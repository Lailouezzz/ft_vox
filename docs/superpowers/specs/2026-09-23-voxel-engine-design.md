# ft_vox — moteur voxel Zig + Vulkan (design)

Date : 2026-09-23 · Branche : `ai` · Zig 0.16.0

## Révisions

- **r2 (2026-09-23)** — après le cœur CPU (jalons 0–2) et les prototypes
  Vulkan / ChunkManager :
  - vulkan-zig passe sur la branche `zig-0.16-compat` (la version épinglée
    vise Zig 0.17-dev et ne compile pas avec 0.16.0) ;
  - cible Vulkan **1.4** (GPU de dev : AMD Radeon Renoir intégré, RADV,
    Vulkan 1.4, pas de mesh shaders) ; SPIR-V passé aux pipelines via
    maintenance5, sans `VkShaderModule` ;
  - aucun descriptor pour la géométrie : tous les buffers passent par leur
    adresse (buffer device address) dans les push constants, dont un
    buffer `FrameData` par frame en vol ;
  - projection perspective infinie en reverse-Z ;
  - métadonnées de chunks mises à jour par `vkCmdUpdateBuffer` dans le
    command buffer (pas d'écriture CPU pendant qu'une frame en vol les lit) ;
  - API du ChunkManager alignée sur le prototype (voir la section
    Streaming) ;
  - `FreeList` : une plage de longueur 0 est une opération nulle (meshes
    vides) ; contrat d'erreur de `WorkerPool.TaskFn` documenté ;
  - mesher : mémoire de travail réutilisée par thread (`threadlocal`).
- **r3 (2026-09-23)** — après les jalons 3–5, relectures et profilage :
  - pools de workers **non** surdimensionnées : `n = cœurs - 1` workers au
    total, `mesh = max(1, n / 3)`, `gen = n - mesh`. Mesuré au chargement :
    10 fps avec 7 + 7 workers sur 8 cœurs (le thread de rendu est affamé),
    110–140 fps avec 5 + 2 ;
  - budget de premiers maillages lancés par `update` (64 ; les remaillages
    après édition ne sont pas limités), `buildVolume` par copies de lignes,
    chunk vide calculé par le worker de génération (profil : 77 % du thread
    principal dans `buildVolume` pendant le chargement) ;
  - ombres : 3 cascades (20 / 64 / 180 blocs), shadow map lue via un push
    descriptor (Vulkan 1.4), culling par vue (caméra + 3 cascades, faces
    tournées vers la lumière pour les cascades) ;
  - réglages en ligne de commande (`--seed`, `--radius`, `--shadow-res`,
    `--day-length`) ;
  - contour du bloc visé : pipeline `line_list` de 24 sommets ;
  - consolidation issue des relectures : recréation atomique de la
    swapchain, `errdefer` d'initialisation, barrière de profondeur
    early+late, pas de recréation en boucle si la taille diffère,
    `WorkerPool.spawnOne` n'enregistre un worker qu'une fois lancé, petits
    correctifs du ChunkManager.
- **r4 (2026-09-24)** — retours d'usage et relectures des jalons 6–9 :
  - T-jonctions du greedy meshing (pixels de ciel visibles entre deux
    blocs) corrigées : chaque quad est élargi d'environ 1/1000 de bloc sur
    ses deux axes dans le vertex shader ;
  - clic de casse : GLFW en mode « sticky mouse buttons », pour qu'un clic
    plus court qu'une frame ne soit pas perdu ;
  - culling des cascades sans plan proche (le depth clamp garde les
    obstacles entre le soleil et la cascade) ; `--day-length` refuse les
    valeurs non finies ;
  - rayon maximal ramené à 24 chunks (marge sur les 16 384 slots et
    16 M quads) ; buffer plein : log, le chunk reste non affiché jusqu'à son
    rechargement ;
  - `n = max(2, cœurs - 1)` workers (au moins un par pool) ;
  - vérifications `comptime` des tailles et offsets des miroirs GPU ;
    format de surface sRGB préféré ; device sans `VK_KHR_swapchain` ou sans
    format/mode de présentation ignoré.
- **r5 (2026-09-24)** — eau transparente (section « Eau transparente »,
  jalons W1–W3) : l'eau sort du hors-périmètre.

## Objectif

Un moteur voxel vitrine, moderne et optimisé, en Zig et Vulkan 1.4 :
monde infini généré procéduralement (relief, grottes, eau, plages, arbres),
binary greedy meshing sur deux `WorkerPool` (génération et maillage), rendu piloté par le GPU
en un seul draw indirect, soleil avec cycle jour/nuit et cascaded shadow maps,
casse de blocs avec remaillage regroupé.

Hors périmètre : pose de blocs, occlusion ambiante,
texte à l'écran, collisions/physique, sauvegarde sur disque, shaders écrits
en Zig (backend SPIR-V de Zig 0.16 insuffisant, testé).

## Contraintes et environnement

- Zig 0.16.0, dépendances : `vulkan-zig` (branche `zig-0.16-compat`,
  commit `b496a6a`) + `vulkan_headers`, `zglfw` (`import_vulkan = true`),
  `zmath`, `znoise` (la lib C FastNoiseLite est liée au module `world`).
- Shaders GLSL compilés en SPIR-V par `glslc` (système) via
  `b.addSystemCommand`, embarqués avec `@embedFile`.
- Vulkan 1.4 requis avec : dynamic rendering, synchronization2,
  maintenance4/5, push descriptors, buffer device address, scalar block
  layout, timeline semaphores, `drawIndirectCount`, `multiDrawIndirect`,
  `drawIndirectFirstInstance`, `shaderInt64`, `depthClamp`. Sinon : le GPU
  est ignoré avec un log qui nomme la fonctionnalité manquante, et le
  programme s'arrête si aucun GPU ne convient.
- Validation layers activées en Debug si présentes.

## Architecture

```
src/
  main.zig              init fenêtre + renderer, boucle, input
  Camera.zig            position, yaw/pitch, view/proj, frustum
  ChunkManager.zig      streaming, états des chunks, jobs, édition
  threading/            module existant : WorkQueue, WorkerPool
  world/                module pur CPU (aucune dépendance GPU/threads)
    root.zig
    block.zig           enum Block + table de couleurs
    Chunk.zig           32³ blocs, ChunkPos, conversions
    terrain.zig         heightAt, isCave, generate
    trees.zig           placement déterministe, débordement inter-chunks
    mesher.zig          binary greedy meshing → quads u64
    raycast.zig         DDA Amanatides & Woo
    FreeList.zig        allocateur de plages (utilisé par le renderer)
  render/
    Context.zig         instance, messenger de debug, surface, device, queue
    Swapchain.zig       swapchain, vues, sémaphores par image
    Buffer.zig          buffer + mémoire + adresse + mapping
    Image.zig           image + mémoire + vue
    pipeline.zig        création de pipelines (maintenance5), layouts
    gpu.zig             miroirs CPU des structures GPU, plans du frustum
    ChunkBuffers.zig    quads, métadonnées, slots, uploads, libérations différées
    Renderer.zig        frames en vol, passes (culling, ombres, scène, ciel)
    shaders/            common.glsl, gpu.glsl, cull.comp, chunk.vert/.frag,
                        shadow.vert, fullscreen.vert, sky.frag,
                        outline.vert/.frag
```

Modules `build.zig` :
- `world` importe `znoise`, `zmath`.
- `threading` n'importe rien.
- Module principal : `world`, `threading`, `vulkan`, `zglfw`, `zmath`.
- Le module `math` vide est supprimé.

`FreeList` vit dans `world` uniquement parce que c'est le module pur testé
sans GPU ; il ne dépend de rien d'autre.

Interfaces :
- `world` : `generate(seed, pos) Chunk`, `mesh(volume *const [34³]Block, out) void`,
  `raycast(origin, dir, max_dist, lookup) ?Hit` où `lookup` renvoie le
  bloc à une position monde (`Hit` = bloc touché + face d'entrée).
- `ChunkManager` : `update(camera_pos)`, `breakBlock(world_pos)`,
  `takeUploads()`, `takeFrees()`. Aucun appel Vulkan.
- `Renderer` : `upload(pos, mesh) !void`, `remove(pos)`,
  `drawFrame(extent, FrameInput) !void` (`FrameInput` : caméra, soleil,
  bloc visé).

## Données du monde

- `Block = enum(u8) { air, grass, dirt, stone, sand, water, log, leaves }`.
- `Chunk` : `blocks: [32768]Block` (32 Ko), index `x + 32·z + 1024·y`,
  méthode `isEmpty()` (tout air → pas de maillage).
- `ChunkPos { x, y, z: i32 }`. Monde → chunk : `>> 5`, local : `& 31`
  (décalage arithmétique, correct en négatif).
- Monde infini en X/Z, `cy ∈ [0, 8)` (256 blocs de haut).

## Génération (fonctions pures)

`generate(seed, pos)` ne dépend que de ses arguments.

1. Relief : FBM 2D znoise → `heightAt(wx, wz)` ∈ ~[40, 140].
2. Couches : surface herbe (sable si hauteur ≤ 64), 3 blocs de terre,
   pierre en dessous.
3. Eau : air sous le niveau de la mer (62) → eau.
4. Grottes : `isCave(wx, wy, wz)` = |bruit 3D| < seuil. Peut ouvrir en
   surface. Jamais dans une colonne dont la surface est ≤ 63.
5. Arbres : hash(seed, wx, wz) < densité, surface herbe, surface non
   creusée. Tronc de 4 à 6 blocs de bois, boule de feuilles de rayon 2.
   Un chunk examine les pieds d'arbres jusqu'à 2 blocs hors de son emprise
   et ne garde que les blocs qui tombent dedans.
6. Graine : `u64` en argument de ligne de commande, valeur par défaut fixe.

## Mesher

- Entrée : volume 34³ (chunk + une couche de chaque voisin de face).
- Binary greedy meshing : une colonne de blocs = un `u64`
  (bit = bloc non-air du type courant). Faces visibles par
  `col & ~(col >> 1)` / `col & ~(col << 1)` contre un masque d'occupation
  opaque, fusion en largeur puis hauteur avec `@ctz` et des ET binaires,
  un jeu de masques par type de bloc.
- Faces des blocs solides émises contre l'air et l'eau ; faces de l'eau
  émises seulement contre l'air.
- Quad `u64` : x, y, z (3 × 6 bits), largeur, hauteur (2 × 6 bits),
  face (3 bits), type de bloc (8 bits).
- Sortie rangée par direction de face : 6 plages par chunk.
- T-jonctions : là où un sommet d'un quad tombe au milieu de l'arête d'un
  quad voisin, les arrondis de rastérisation peuvent laisser un pixel non
  couvert. Le vertex shader élargit chaque quad opaque d'environ 1/1000 de
  bloc sur ses deux axes ; le chevauchement est coplanaire et invisible.
  Les quads d'eau ne sont pas élargis : l'eau est mélangée sans écrire la
  profondeur, un chevauchement serait mélangé deux fois (lignes visibles),
  alors qu'une fissure ne laisse voir que le fond.
- `zig build bench` : temps de maillage par chunk, cible < 200 µs en
  ReleaseFast.

## Streaming et édition (ChunkManager)

Deux `WorkerPool` typées, reliées par le ChunkManager (4 files) :

```
ChunkPos ─gen_in─► GenPool ─gen_out─► ChunkManager ─mesh_in─► MeshPool ─mesh_out─► ChunkManager ─► GPU
                                      (6 voisins prêts,   ▲
                                       snapshot 34³)      └─ remaillage après édition : pushFront
```

- `GenPool = WorkerPool(GenJob, GenResult)` : `GenJob = { id, pos, seed,
  chunk: *Chunk }` (le chunk est alloué par l'émetteur), `GenResult = { id,
  pos, chunk: *Chunk }`. La génération ne peut pas échouer.
- `MeshPool = WorkerPool(MeshJob, MeshResult)` : `MeshJob = { id, pos,
  version, volume: *Volume }`, `MeshResult = { id, pos, version, mesh:
  ?Mesh }` (`null` = échec, le chunk redevient `dirty`). La tâche attrape
  ses erreurs (une erreur renvoyée ferait perdre l'item) et libère elle-même
  le volume.
- Pas de pipe direct : un chunk généré ne peut être maillé qu'une fois ses
  6 voisins générés, c'est le ChunkManager (thread principal) qui fait le
  lien.
- Nombre de workers de chaque pool fourni par l'appelant via `Config`.
  `main` en lance `n = max(2, cœurs - 1)` au total (un cœur reste au rendu) :
  `mesh = max(1, n / 3)`, `gen = n - mesh`. Surdimensionner les deux pools
  affame le thread de rendu pendant le chargement (mesuré : 10 fps).
- Résultats lus sans bloquer avec `pop()` sur `gen_out` et `mesh_out`.

API : `create(gpa, io, Config) !*ChunkManager` / `destroy()`,
`update(center: BlockPos) !void`, `takeUploads() []const Upload`
(`Upload = { pos, mesh }`), `takeUnloads() []const ChunkPos` (listes
valides jusqu'au prochain `update`, les meshes sont libérés par le
ChunkManager), `breakBlock(pos) !bool`, `blockAt(pos) Block` (le
ChunkManager sert de `lookup` au raycast), `stats`. Le renderer applique les
uploads avant les unloads (une même position peut apparaître dans les deux
la même frame). `Config = { seed, radius = 16, unload_margin = 2,
gen_workers, mesh_workers }` ; rayon euclidien horizontal en chunks.

Par chunk (`HashMap(ChunkPos, Entry)`) :
- état `generating` ou `generated: *Chunk` ;
- `id` unique (détecte un résultat pour un chunk déchargé puis rechargé) ;
- `version: u32` (commence à 1, incrémentée à chaque édition), `dirty`,
  `mesh_in_flight`, `displayed_version` (0 = rien d'affiché). Pas
  d'allocation GPU : le renderer indexe ses allocations par `ChunkPos`.

Chaque `update` :
1. Libère les uploads/unloads de la frame précédente.
2. Résultats : un mesh plus récent que `displayed_version` devient un
   upload, sinon il est libéré ; `id` inconnu : libéré ;
   `mesh_in_flight = false`.
3. Pour chaque chunk `dirty`, pas `mesh_in_flight`, dans le rayon et dont
   les 6 voisins sont générés (hors de `[0, 8)` en Y = air) : copie du
   volume 34³ (39 Ko) dans le job, `dirty = false`, `mesh_in_flight = true`.
   Les chunks entièrement vides (drapeau calculé par le worker de
   génération) ne sont pas maillés. Au plus `max_mesh_dispatch` (64)
   premiers maillages par `update` ; les remaillages après édition
   (`version > 1`) ne sont pas limités et passent en tête de `mesh_in`.
   `buildVolume` copie par lignes de 32 blocs.
4. Quand la colonne centrale change : génération des chunks manquants de
   l'anneau rayon + 1, du plus proche au plus loin (chunks édités restaurés
   depuis la table des chunks édités), puis déchargement au-delà de
   rayon + `unload_margin`.

Édition : `breakBlock` (thread principal seulement) met le bloc à `air`,
`version += 1`, `dirty = true`, et marque le voisin de face si la coordonnée
locale est 0 ou 31. Renvoie `!bool` : l'ajout à la table des chunks édités
peut manquer de mémoire, et il est fait avant de modifier le bloc. Le chunk est copié dans la table des chunks édités.
Garantie : au plus un maillage en cours et un en attente par chunk, donc
N éditions pendant un maillage → exactement un remaillage.
Limite connue : la table des chunks édités grossit sans fin.

## Rendu (Vulkan 1.4)

Frames : 2 en vol ; un sémaphore « rendu terminé » par image de la
swapchain ; swapchain recréée sur resize / `OUT_OF_DATE` / `SUBOPTIMAL` ;
mode de présentation mailbox si disponible, sinon FIFO.

Caméra : perspective infinie en reverse-Z (profondeur 1 au plan proche, 0 à
l'infini, test `GREATER_OR_EQUAL`, clear à 0) ; matrices zmath en
convention vecteur-ligne, dont la disposition mémoire est celle attendue
par `mat4 * vec4` en GLSL.

Pipelines : dynamic rendering, SPIR-V passé directement aux étages via
maintenance5 (pas de `VkShaderModule`), viewport/scissor dynamiques.

Géométrie pilotée par le GPU, sans descriptor :
1. Buffer de quads de 128 Mo (device local, 16 M quads), découpé par
   `FreeList`. Uploads via un staging buffer par frame en vol ; ce qui ne
   tient pas dans le budget de la frame est copié et reporté.
2. Buffer de métadonnées (`ChunkMeta` : origine, premier quad, 6 comptes,
   actif) indexé par slot, 16 384 slots ; mis à jour par
   `vkCmdUpdateBuffer` dans le command buffer de la frame. Chaque frame
   commence par une barrière qui ordonne toutes les lectures GPU des frames
   précédentes (culling, draw indirect, vertex pulling) avant ses écritures :
   slots et plages libérés sont donc réutilisables tout de suite.
3. `FrameData` par frame en vol (host visible) : view-proj, plans du
   frustum, position caméra, soleil, ambiance, brouillard, palette des
   blocs (source unique : `Block.color`), capacité.
4. `cull.comp`, une invocation par slot : frustum culling + élimination des
   directions de face invisibles depuis le point de vue ; écrit des
   `VkDrawIndirectCommand` (`firstInstance = slot·8 + face`) avec un
   compteur atomique.
5. Un `vkCmdDrawIndirectCount` pour tout le monde.
6. `chunk.vert` lit quads et métadonnées par adresse de buffer et
   reconstruit les 6 sommets depuis `gl_VertexIndex`.
7. Push constants communs aux pipelines de chunks : adresses de
   `FrameData`, métadonnées, quads, commandes indirectes, compteur.

Soleil et ombres :
- 3 cascades (jusqu'à 20, 64 et 180 blocs de la caméra), tableau de
  profondeur D32 d'une couche par cascade, résolution réglable (2048 par
  défaut).
- Culling par vue : `cull.comp` tourne pour la caméra puis pour chaque
  cascade (plans du frustum de la cascade, seules les faces tournées vers
  la lumière) ; zone de commandes indirectes et compteur par vue.
- Chaque cascade est dessinée par le même chemin indirect avec
  `shadow.vert` (profondeur seule, depth clamp, depth bias négatif en
  reverse-Z).
- Cascade ajustée par sphère englobante de la tranche de frustum (stable en
  rotation) et projection orthographique reverse-Z calée sur la grille de
  texels ; décalage selon la normale d'environ 1,5 texel ; PCF 3×3 avec un
  sampler de comparaison `GREATER_OR_EQUAL`, bord à 0 (éclairé).
- La shadow map est liée par un push descriptor (pas de pool ni de sets).
- Cycle jour/nuit à durée réglable ; lumière de lune faible et ombres
  coupées quand le soleil est sous l'horizon.

Ciel et finition : triangle plein écran, dégradé analytique selon
l'élévation du soleil + disque solaire ; brouillard vers la couleur du ciel
en bord de distance de rendu ; tone mapping ACES.

Casse : raycast DDA depuis la caméra à chaque frame (le ChunkManager sert
de `lookup`), portée 8 blocs ; un clic gauche (front montant) casse le bloc
visé (GLFW en mode « sticky mouse buttons » : un clic plus court qu'une
frame n'est pas perdu) ; contour fil de fer : pipeline `line_list` de 24 sommets générés dans
le vertex shader, légèrement agrandi, test de profondeur sans écriture.

Réglages : `ft_vox [--seed N] [--radius 4..24] [--shadow-res 512..4096,
puissance de 2] [--day-length SECONDES]` ; valeurs par défaut pour iGPU
(rayon 16, ombres 2048, journée de 240 s) ; une option invalide affiche
l'usage et quitte avec le code 2.

Titre de fenêtre : FPS, chunks affichés, quads résidents sur le GPU,
position.

Contrôles : vol libre, ZQSD + souris, Shift pour accélérer, Échap quitte.

## Eau transparente (r5)

Rendu : eau transparente avec reflet du ciel (Fresnel), surface abaissée,
vue sous l'eau, ombres reçues, opacité selon la profondeur. Hors
périmètre : vagues, vrais reflets du terrain, tri des faces d'eau,
simulation de fluide (l'eau reste statique, niveau de la mer fixe).

Mesher :
- 12 groupes de quads par chunk : 6 directions opaques puis 6 directions
  d'eau, dans l'ordre de `Face` ; `Mesh.counts: [12]u32`.
- En interne, deux types d'eau qui ne fusionnent pas : « surface » (bloc
  d'eau dont le bloc au-dessus n'est pas de l'eau) et « profonde ». Les
  quads d'eau de surface ont le bit 41 du quad à 1 (champ `surface` pris sur
  le padding) ; le type émis reste `water`.
- Le vertex shader abaisse de 1/8 de bloc le haut des quads marqués : toute
  la face +Y, et l'arête haute des faces latérales. Les faces solides contre
  l'eau étant émises, aucun trou n'apparaît sous la surface abaissée.

Données GPU :
- `ChunkMeta.counts: [12]u32` (68 octets) ; vérifications `comptime` mises
  à jour.
- `firstInstance = slot·16 + groupe` (0–5 opaque, 6–11 eau).
- Culling caméra : groupes opaques dans la plage de la vue 0 (règle de
  visibilité par face inchangée), groupes d'eau dans une 5e plage (vue
  « eau », compteur 5) sans élimination par direction de face (l'eau se
  voit aussi par dessous). Cascades : groupes d'eau ignorés (l'eau ne
  projette pas d'ombre).

Rendu (dynamic rendering local read, Vulkan 1.4, `dynamicRenderingLocalRead`) :
- Une seule passe : ciel, chunks opaques, barrière par région à
  l'intérieur du rendu (écritures de profondeur → lecture en input
  attachment par le fragment shader), eau, contour du bloc visé.
- L'image de profondeur a l'usage `input_attachment` et reste en layout
  `RENDERING_LOCAL_READ` pendant la passe ; le pipeline de l'eau déclare la
  profondeur comme input attachment (`RenderingInputAttachmentIndexInfo`)
  et la lit avec `subpassLoad`.
- Pipeline de l'eau : mélange alpha (src alpha / 1 - src alpha), test de
  profondeur `GREATER_OR_EQUAL` sans écriture, pas de culling de faces.
- `water.frag` : épaisseur d'eau = distance au fond (depth buffer,
  linéarisée depuis le reverse-Z infini) - distance à la surface ; opacité
  et teinte croissent avec l'épaisseur ; reflet du ciel pondéré par Fresnel
  (Schlick, F0 = 0,02) ; reflet spéculaire du soleil ; ombres reçues. Le
  calcul d'ombre et d'éclairage passe dans `lighting.glsl`, partagé avec
  `chunk.frag`.
- Repli si la validation refuse le local read : couper la passe, copier la
  profondeur, relancer un rendu pour l'eau (révision de spec avant de le
  faire).

Sous l'eau : `main` interroge `ChunkManager.blockAt(position caméra)` ; si
ce bloc est de l'eau de surface (pas d'eau au-dessus), la caméra doit aussi
être sous la surface abaissée (y < haut du bloc - 1/8). `underwater` est
passé au renderer ; brouillard bleu serré (fin à ~24 blocs) et
teinte appliqués au ciel, aux chunks et à l'eau ; la surface vue par
dessous utilise la normale retournée (`gl_FrontFacing`), et son épaisseur
d'eau est la distance caméra-surface (ce qu'il y a derrière est de l'air).

Tests :
- mesher : faces +Y d'eau marquées ; eau profonde non marquée ; surface et
  profonde non fusionnées ; faces latérales du bloc de surface marquées ;
  test de propriété (couverture = mesher naïf) inchangé.
- GPU : `comptime` sur `ChunkMeta` (68 octets) et le nouvel encodage.
- visuel (`xdotool` + captures) : lac vu d'en haut (fond visible près des
  rives, plus sombre au large), vue rasante (reflet du ciel), caméra sous
  l'eau (brouillard bleu), ombres des arbres sur l'eau ; validation sans
  aucun message ; au moins 95 fps sur la scène de référence (106 avant).

Jalons :
- W1 : mesher (groupes d'eau, bit de surface) et données GPU (12 comptes,
  5e plage) ; l'eau est encore dessinée opaque, surface abaissée visible.
- W2 : passe transparente en local read, profondeur, Fresnel, ombres,
  `lighting.glsl`.
- W3 : vue sous l'eau.

## Gestion des erreurs

- Erreur Vulkan : remontée jusqu'à `main`, log, sortie propre.
- `OUT_OF_DATE` / `SUBOPTIMAL` : recréation de la swapchain.
- Buffer de quads ou slots pleins : log, le chunk reste non affiché jusqu'à
  son rechargement, pas de crash (le rayon maximal de 24 garde une marge).
- Erreur dans un worker (OOM) : log, le job est relancé plus tard.
- Fermeture : `shutdown` de la GenPool puis de la MeshPool, les résultats
  restants sont libérés.

## Tests (`zig build test`, sans GPU)

- world : déterminisme ; coordonnées -1 / 31 / 32 ; ordre des couches ;
  eau sous le niveau de la mer seulement ; pas de grotte sous une colonne
  immergée ; arbre à cheval sur une bordure identique des deux côtés.
- mesher : 1 bloc → 6 quads ; cube 32³ plein → 6 quads ; types différents
  non fusionnés ; bordure voisine masque les faces ; aller-retour
  d'encodage `u64` ; propriété : sur des volumes aléatoires, les faces
  couvertes par les quads = les faces d'un mesher naïf de référence.
- raycast : axes, diagonales, négatifs, portée max.
- FreeList : alloc, free, fusion de voisins, plein.
- ChunkManager : ensemble voulu, transitions d'état, hystérésis, priorité des remaillages dans `mesh_in` ;
  N éditions pendant un maillage → 1 remaillage ; résultat dépassé envoyé
  s'il est plus récent que l'affiché, jeté sinon ; bordure marque le
  voisin ; chunk édité déchargé puis rechargé garde ses modifications.
- rendu (sans GPU) : projection reverse-Z et axe Y de Vulkan ; plans du
  frustum ; ajustement des cascades (le point devant la caméra tombe dans
  chaque cascade, plus près du soleil = profondeur plus grande) ; cycle du
  soleil.
- réglages : valeurs par défaut, toutes les options, erreurs (option
  inconnue, valeur manquante, hors bornes).
- rendu (avec GPU, vérification manuelle ou scriptée avec `xdotool` et
  des captures) : lancement avec les validation layers, aucun message
  `error(vulkan)` ni `warning(vulkan)`.

## Jalons

Chacun compile, passe ses tests, et fait l'objet d'un commit.

0. Ajustements `threading` :
   - `WorkerPool` : une erreur de tâche est loggée et le worker continue
     (seul `error.Canceled` remonte) ;
   - `WorkQueue.pop()` : `error.Closed` seulement si fermée **et** vide,
     comme `waitPop` ;
   - `drain()` conservé, doc comment : les items sont jetés sans être
     libérés ;
   - tests : une tâche qui échoue ne fait perdre aucun worker ;
     `pop()` après `close()` rend les items restants puis `error.Closed`.
1. `world` : blocs, chunk, génération (relief, grottes, eau, arbres),
   raycast, FreeList + tests.
2. `mesher` : binary greedy meshing + tests + bench.
3. Bootstrap Vulkan 1.4 : passage de vulkan-zig sur `zig-0.16-compat`,
   compilation des shaders, fenêtre, contexte, swapchain, dynamic
   rendering, caméra reverse-Z, ciel.
4. ChunkManager : deux pools, streaming, édition, tests (CPU, sans GPU ;
   peut avancer en parallèle du jalon 3).
5. Chunks à l'écran, pilotés par le GPU : `ChunkBuffers`, `FrameData`,
   `cull.comp` + `drawIndirectCount`, vertex pulling, éclairage soleil +
   ambiance, brouillard, tone mapping, contrôles caméra, branchement du
   ChunkManager.
6. Consolidation : correctifs des relectures (swapchain, errdefer,
   barrières, ChunkManager, WorkerPool) et performance du chargement
   (répartition des workers, budget de maillage, `buildVolume` par lignes).
7. Cascaded shadow maps + cycle jour/nuit (déjà en place depuis le
   jalon 5 pour la lumière).
8. Réglages en ligne de commande.
9. Casse de blocs : raycast caméra, contour, remaillage regroupé.
