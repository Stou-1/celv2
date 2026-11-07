# Freely licensed sources for training and benching (2026-11-02)

Candidates for training data of encoder decision models (kept apart from the 8 bench sources, which are
not freely licensed). Nothing downloaded yet; check the licence file shipped with each before use.

| Source | Kind | Licence | Form | Where | Notes |
|---|---|---|---|---|---|
| Sol Levante (Netflix Open Content) | 2D anime (Production I.G) | CC BY 4.0 | 4K HDR PQ masters, TGA, PSD | https://opencontent.netflix.com/ | HDR: tone-map to SDR 8-bit for now; the closest to the anime sources |
| Sita Sings the Blues (xiph derf, 1080p y4m) | 2D flat animation | CC0 (visuals; the 1920s songs excluded) | 117,714 frames, 341 GB xz | https://media.xiph.org/video/derf/ | Flat colours like cel anime; take a few scenes |
| Morevna episodes 3, 4 (Internet Archive, Gumroad) | 2D anime-style vector animation | CC BY-SA 3.0 / 4.0 | 1080p | https://archive.org/details/morevna-ep3 (ep. 3), https://morevnaproject.gumroad.com/l/morevna-episode-4 (ep. 4) | Share-alike: derived model weights should be checked against it |
| Blender Studio films (Sintel trailer, Sprite Fright, Spring, Big Buck Bunny, ...) | 3D animation | CC BY (credit "(CC) Blender Foundation \| studio.blender.org") | 1080p y4m on xiph derf for the older ones | https://studio.blender.org/films/ , https://media.xiph.org/video/derf/ | Logos excluded |
| Cosmos Laundromat (Netflix Open Content) | 3D animation | CC BY 4.0 | 2K HDR, EXR | https://opencontent.netflix.com/ | |
| SVT MultiFormat (crowd_run, ducks_take_off, in_to_tree, old_town_cross, park_joy) | live action | per the SVT readme (free for research) | 1080p / 2160p50 y4m, 500 frames | https://media.xiph.org/video/derf/ | Readme terms to be read before training use |
| Netflix El Fuente, Chimera, Meridian, Sparks, Nocturne | live action | CC BY 4.0 (Netflix Open Content) | 4K 10-bit y4m / TIFF / EXR | https://opencontent.netflix.com/ (El Fuente and Chimera y4m also on https://media.xiph.org/video/derf/) | |

Sources: opencontent.netflix.com, media.xiph.org/video/derf, studio.blender.org (licensing pages),
morevnaproject.org, blog.ninapaley.com.
