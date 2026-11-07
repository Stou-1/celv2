#include "core/film_grain.h"

#include <stdexcept>
#include <vector>

extern "C" {
#include "avm_dsp/noise_model.h"
#include "avm_dsp/noise_util.h"
}

namespace {

constexpr int kBlock = 32;  // aomenc's default --denoise-block-size

}  // namespace

struct GrainModel::Model {
  avm_noise_model_t model{};
  bool ready = false;
};

struct GrainModel::Denoised {
  int w = 0, h = 0;  // the padded frame
  std::vector<uint16_t> raw[3], clean[3];
  std::vector<uint8_t> flat;
};

GrainModel::GrainModel(int bit_depth, int level) : m_(std::make_unique<Model>()), bit_depth_(bit_depth), level_(level / 10.0f) {
  const avm_noise_model_params_t params = {AVM_NOISE_SHAPE_SQUARE, 3, bit_depth};  // as avm_denoise_and_model_run()
  if (!avm_noise_model_init(&m_->model, params)) throw std::runtime_error("avm_noise_model_init failed");
  m_->ready = true;
}

GrainModel::~GrainModel() {
  if (m_->ready) avm_noise_model_free(&m_->model);
}

std::shared_ptr<GrainModel::Denoised> GrainModel::denoise(const Frame420& f) const {
  auto d = std::make_shared<Denoised>();
  // AVM's buffers hold 16-bit samples: the padded frame.
  d->w = f.pw, d->h = f.ph;
  const int cw = d->w / 2;
  if (f.bit_depth > 8) {
    d->raw[0] = f.hy, d->raw[1] = f.hu, d->raw[2] = f.hv;
  } else {
    d->raw[0].assign(f.y.begin(), f.y.end()), d->raw[1].assign(f.u.begin(), f.u.end()), d->raw[2].assign(f.v.begin(), f.v.end());
  }
  for (int p = 0; p < 3; p++) d->clean[p].resize(d->raw[p].size());
  avm_flat_block_finder_t finder;
  if (!avm_flat_block_finder_init(&finder, kBlock, bit_depth_)) throw std::runtime_error("avm_flat_block_finder_init failed");
  d->flat.resize(static_cast<size_t>((d->w + kBlock - 1) / kBlock) * ((d->h + kBlock - 1) / kBlock));
  avm_flat_block_finder_run(&finder, reinterpret_cast<const uint8_t*>(d->raw[0].data()), d->w, d->h, d->w, d->flat.data());
  avm_flat_block_finder_free(&finder);
  // A flat noise spectrum, as avm_denoise_and_model_run() uses.
  std::vector<float> psd_y(kBlock * kBlock, avm_noise_psd_get_default_value(kBlock, level_));
  std::vector<float> psd_uv(kBlock * kBlock, avm_noise_psd_get_default_value(kBlock >> 1, level_));
  const uint8_t* const data[3] = {reinterpret_cast<const uint8_t*>(d->raw[0].data()), reinterpret_cast<const uint8_t*>(d->raw[1].data()),
                                  reinterpret_cast<const uint8_t*>(d->raw[2].data())};
  uint8_t* clean[3] = {reinterpret_cast<uint8_t*>(d->clean[0].data()), reinterpret_cast<uint8_t*>(d->clean[1].data()),
                       reinterpret_cast<uint8_t*>(d->clean[2].data())};
  int strides[3] = {d->w, cw, cw}, sub[2] = {1, 1};
  float* psd[3] = {psd_y.data(), psd_uv.data(), psd_uv.data()};
  if (!avm_wiener_denoise_2d(data, clean, d->w, d->h, strides, sub, psd, kBlock, bit_depth_))
    throw std::runtime_error("AVM's Wiener denoising failed");
  return d;
}

void GrainModel::finish(Frame420& f, Denoised& d) {
  const uint8_t* const data[3] = {reinterpret_cast<const uint8_t*>(d.raw[0].data()), reinterpret_cast<const uint8_t*>(d.raw[1].data()),
                                  reinterpret_cast<const uint8_t*>(d.raw[2].data())};
  const uint8_t* const clean[3] = {reinterpret_cast<const uint8_t*>(d.clean[0].data()), reinterpret_cast<const uint8_t*>(d.clean[1].data()),
                                   reinterpret_cast<const uint8_t*>(d.clean[2].data())};
  int strides[3] = {d.w, d.w / 2, d.w / 2}, sub[2] = {1, 1};
  // avm_denoise_and_model_run(): a frame of another noise type starts the model afresh; without an
  // update the last estimate stands.
  const avm_noise_status_t status = avm_noise_model_update(&m_->model, data, clean, d.w, d.h, strides, sub, d.flat.data(), kBlock);
  bool have = status == AVM_NOISE_STATUS_OK;
  if (status == AVM_NOISE_STATUS_DIFFERENT_NOISE_TYPE) avm_noise_model_save_latest(&m_->model), have = true;
  if (!have) have = m_->model.combined_state[0].strength_solver.num_equations > 0;
  f.grain.reset();
  if (!have) return;
  avm_film_grain_t g{};
  g.bit_depth = static_cast<unsigned>(bit_depth_);
  if (!avm_noise_model_get_grain_parameters(&m_->model, &g)) throw std::runtime_error("AVM's grain parameters failed");
  if (f.bit_depth > 8) {
    f.hy = std::move(d.clean[0]), f.hu = std::move(d.clean[1]), f.hv = std::move(d.clean[2]);
  } else {
    for (size_t i = 0; i < d.clean[0].size(); i++) f.y[i] = static_cast<uint8_t>(d.clean[0][i]);
    for (size_t i = 0; i < d.clean[1].size(); i++) f.u[i] = static_cast<uint8_t>(d.clean[1][i]), f.v[i] = static_cast<uint8_t>(d.clean[2][i]);
  }
  pad_frame(f);
  round_to_8bit(f);
  auto p = std::make_shared<GrainParams>();
  p->p = g;
  f.grain = std::move(p);
}

bool same_grain_model(const avm_film_grain_t& a, const avm_film_grain_t& b) {
  return av2_check_grain_params_equiv(&a, &b) != 0;
}

uint16_t grain_seed(int display) {
  const uint16_t s = static_cast<uint16_t>(7391 + 3381 * static_cast<unsigned>(display));  // libaom's step
  return s ? s : 1;
}
