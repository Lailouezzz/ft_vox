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

## Objectif

Un moteur voxel vitrine, moderne et optimisé, en Zig et Vulkan 1.4 :
monde infini généré procéduralement (relief, grottes, eau, plages, arbres),
binary greedy meshing sur deux `WorkerPool` (génération et maillage), rendu piloté par le GPU
en un seul draw indirect, soleil avec cycle jour/nuit et cascaded shadow maps,
casse de blocs avec remaillage regroupé.

Hors périmètre : pose de blocs, transparence de l'eau, occlusion ambiante,
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
- Nombre de workers de chaque pool fourni par l'appelant via `Config`
  (`max(1, cœurs - 1)` chacune dans `main`) : un worker inactif dort dans
  `waitPop`, donc une pool seule occupe tous les cœurs.
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
   Les chunks entièrement vides ne sont pas maillés. Remaillage après
   édition (`version > 1`) en tête de `mesh_in`, premier maillage en queue.
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
   `vkCmdUpdateBuffer` dans le command buffer de la frame. Slots et plages
   libérés seulement quand plus aucune frame en vol ne les utilise.
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
- 3 cascades, chacune culled et dessinée par le même chemin indirect avec
  le frustum du soleil.
- Matrices calées sur la grille de texels, PCF 3×3.
- Cycle jour/nuit à durée réglable ; lumière de lune faible et ombres
  coupées quand le soleil est sous l'horizon.

Ciel et finition : triangle plein écran, dégradé analytique selon
l'élévation du soleil + disque solaire ; brouillard vers la couleur du ciel
en bord de distance de rendu ; tone mapping ACES.

Casse : raycast DDA depuis la caméra, portée 8 blocs, clic gauche casse,
contour fil de fer (24 sommets) sur le bloc visé.

Titre de fenêtre : FPS, chunks chargés, quads dessinés.

Contrôles : vol libre, ZQSD + souris, Shift pour accélérer, Échap quitte.

## Gestion des erreurs

- Erreur Vulkan : remontée jusqu'à `main`, log, sortie propre.
- `OUT_OF_DATE` / `SUBOPTIMAL` : recréation de la swapchain.
- Buffer de quads plein : log, plus de nouveaux chunks chargés, pas de crash.
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
6. Cascaded shadow maps + cycle jour/nuit.
7. Finitions : stats dans le titre, réglages de qualité pour iGPU
   (distance de rendu, résolution des ombres) en arguments.
8. Casse de blocs : raycast caméra, contour, remaillage regroupé.
