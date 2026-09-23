# ft_vox — moteur voxel Zig + Vulkan (design)

Date : 2026-09-23 · Branche : `ai` · Zig 0.16.0

## Objectif

Un moteur voxel vitrine, moderne et optimisé, en Zig et Vulkan 1.3 :
monde infini généré procéduralement (relief, grottes, eau, plages, arbres),
binary greedy meshing sur la `WorkerPool` existante, rendu piloté par le GPU
en un seul draw indirect, soleil avec cycle jour/nuit et cascaded shadow maps,
casse de blocs avec remaillage regroupé.

Hors périmètre : pose de blocs, transparence de l'eau, occlusion ambiante,
texte à l'écran, collisions/physique, sauvegarde sur disque, shaders écrits
en Zig (backend SPIR-V de Zig 0.16 insuffisant, testé).

## Contraintes et environnement

- Zig 0.16.0, dépendances existantes : `vulkan-zig` + `vulkan_headers`,
  `zglfw` (`import_vulkan = true`), `zmath`, `znoise`.
- Shaders GLSL compilés en SPIR-V par `glslc` (système) via
  `b.addSystemCommand`, embarqués avec `@embedFile`.
- Vulkan 1.3 requis avec : dynamic rendering, synchronization2,
  buffer device address, `drawIndirectCount`. Sinon : arrêt avec un message
  qui nomme la fonctionnalité manquante.
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
    Context.zig         instance, device, queue, swapchain
    Renderer.zig        frames, passes, buffers, upload
    shaders/            cull.comp, chunk.vert/.frag, shadow.vert,
                        sky.vert/.frag, outline.vert/.frag
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
- `Renderer` : `upload(chunk_id, quads) !Allocation`, `free(Allocation)`,
  `drawFrame(camera, time_of_day, target: ?BlockPos)`.

## Données du monde

- `Block = enum(u8) { air, grass, dirt, stone, sand, water, log, leaves }`.
- `Chunk` : `blocks: [32768]Block` (32 Ko), index `x + 32·z + 1024·y`,
  drapeau `empty` (tout air → pas de maillage).
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

Une `WorkerPool(Job, Result)` unique, `Job = union { generate, mesh }`.
Résultats lus sans bloquer avec `pop()` sur la file de sortie.

Par chunk (`HashMap(ChunkPos, Entry)`) :
- état `generating → generated → meshing → ready` ;
- `id` unique (détecte un résultat pour un chunk déchargé puis rechargé) ;
- `version: u32` (incrémenté à chaque édition), `dirty`, `mesh_in_flight`,
  `displayed_version`, allocation GPU courante.

Chaque frame :
1. Calcul de l'ensemble voulu : rayon horizontal réglable (16 par défaut),
   trié du plus proche au plus loin. Générations manquantes lancées
   (chunks édités rechargés depuis la table des chunks édités).
2. Un chunk est `dirty` dès que ses 6 voisins sont générés (les voisins
   hors de `[0, 8)` en Y comptent comme de l'air).
3. Pour chaque chunk `dirty` et pas `mesh_in_flight` : copie du volume 34³
   (39 Ko) dans le job, `dirty = false`, `mesh_in_flight = true`.
   Les jobs d'édition passent devant avec `pushFront`.
4. Résultats : un mesh plus récent que `displayed_version` est envoyé au
   GPU, sinon il est jeté. `mesh_in_flight = false`. Résultat d'un `id`
   inconnu : jeté.
5. Déchargement au-delà du rayon + 2. Allocation GPU libérée après que
   les frames en vol ne l'utilisent plus.

Édition : `breakBlock` (thread principal seulement) met le bloc à `air`,
`version += 1`, `dirty = true`, et marque le voisin de face si la coordonnée
locale est 0 ou 31. Le chunk est copié dans la table des chunks édités.
Garantie : au plus un maillage en cours et un en attente par chunk, donc
N éditions pendant un maillage → exactement un remaillage.
Limite connue : la table des chunks édités grossit sans fin.

## Rendu (Vulkan 1.3)

Frames : 2 en vol, swapchain recréée sur resize / `OUT_OF_DATE` /
`SUBOPTIMAL`.

Géométrie pilotée par le GPU :
1. Buffer de quads de 256 Mo (device local), découpé par `FreeList`.
   Uploads via un staging buffer par frame en vol.
2. Buffer de métadonnées par chunk : origine + (début, taille) × 6 plages.
3. `cull.comp`, une invocation par chunk : frustum culling + élimination
   des directions de face invisibles depuis le point de vue. Écrit des
   `VkDrawIndirectCommand` avec un compteur atomique.
4. Un `vkCmdDrawIndirectCount` pour tout le monde.
5. `chunk.vert` lit les quads via une adresse de buffer (push constant),
   reconstruit les 6 sommets depuis `gl_VertexIndex`.

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
- Erreur dans un worker (OOM) : log, le chunk est relancé plus tard.

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
- ChunkManager : ensemble voulu, transitions d'état, hystérésis, priorité ;
  N éditions pendant un maillage → 1 remaillage ; résultat dépassé envoyé
  s'il est plus récent que l'affiché, jeté sinon ; bordure marque le
  voisin ; chunk édité déchargé puis rechargé garde ses modifications.

## Jalons

Chacun compile, passe ses tests, et fait l'objet d'un commit.

1. `world` : blocs, chunk, génération (relief, grottes, eau, arbres),
   raycast, FreeList + tests.
2. `mesher` : binary greedy meshing + tests + bench.
3. Bootstrap Vulkan 1.3 : fenêtre, device, swapchain, dynamic rendering,
   ciel.
4. Chunks à l'écran : buffer de quads, vertex pulling, caméra, ChunkManager.
5. Culling GPU + `drawIndirectCount`.
6. Cascaded shadow maps + cycle jour/nuit.
7. Finitions : brouillard, tone mapping, stats dans le titre.
8. Casse de blocs : raycast caméra, contour, remaillage regroupé.
