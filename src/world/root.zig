const coords = @import("coords.zig");

pub const Block = @import("block.zig").Block;
pub const Chunk = @import("Chunk.zig");
pub const ChunkPos = coords.ChunkPos;
pub const BlockPos = coords.BlockPos;
pub const LocalPos = coords.LocalPos;
pub const Face = coords.Face;
pub const chunk_size = coords.chunk_size;
pub const height_chunks = coords.height_chunks;

pub const terrain = @import("terrain.zig");
pub const Generator = terrain.Generator;
pub const generate = terrain.generate;

test {
    _ = @import("block.zig");
    _ = @import("coords.zig");
    _ = @import("Chunk.zig");
    _ = @import("terrain.zig");
}
