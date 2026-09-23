//! First-fit range allocator over [0, capacity). Hands out offsets into one
//! big GPU buffer; knows nothing about the GPU itself.
const std = @import("std");
const Allocator = std.mem.Allocator;

const FreeList = @This();

pub const Range = struct {
    offset: u32,
    len: u32,
};

/// Free ranges, sorted by offset, never adjacent (merged on free).
free_ranges: std.ArrayList(Range),
capacity: u32,

pub fn init(gpa: Allocator, capacity: u32) Allocator.Error!FreeList {
    var free_ranges: std.ArrayList(Range) = .empty;
    try free_ranges.append(gpa, .{ .offset = 0, .len = capacity });
    return .{ .free_ranges = free_ranges, .capacity = capacity };
}

pub fn deinit(fl: *FreeList, gpa: Allocator) void {
    fl.free_ranges.deinit(gpa);
    fl.* = undefined;
}

/// Returns null when no free range is large enough.
pub fn alloc(fl: *FreeList, len: u32) ?Range {
    std.debug.assert(len > 0);
    for (fl.free_ranges.items, 0..) |*r, i| {
        if (r.len < len) continue;
        const out: Range = .{ .offset = r.offset, .len = len };
        if (r.len == len) {
            _ = fl.free_ranges.orderedRemove(i);
        } else {
            r.offset += len;
            r.len -= len;
        }
        return out;
    }
    return null;
}

/// Gives `range` back, merging it with adjacent free ranges.
pub fn free(fl: *FreeList, gpa: Allocator, range: Range) Allocator.Error!void {
    const items = fl.free_ranges.items;
    var i: usize = 0;
    while (i < items.len and items[i].offset < range.offset) : (i += 1) {}

    const merge_prev = i > 0 and items[i - 1].offset + items[i - 1].len == range.offset;
    const merge_next = i < items.len and range.offset + range.len == items[i].offset;
    if (merge_prev and merge_next) {
        items[i - 1].len += range.len + items[i].len;
        _ = fl.free_ranges.orderedRemove(i);
    } else if (merge_prev) {
        items[i - 1].len += range.len;
    } else if (merge_next) {
        items[i].offset = range.offset;
        items[i].len += range.len;
    } else {
        try fl.free_ranges.insert(gpa, i, range);
    }
}

/// Total free units (for stats).
pub fn freeCount(fl: *const FreeList) u32 {
    var n: u32 = 0;
    for (fl.free_ranges.items) |r| n += r.len;
    return n;
}

// ---
// Tests
// ---

const testing = std.testing;

test "alloc until full, then null" {
    var fl: FreeList = try .init(testing.allocator, 10);
    defer fl.deinit(testing.allocator);
    try testing.expectEqual(Range{ .offset = 0, .len = 4 }, fl.alloc(4).?);
    try testing.expectEqual(Range{ .offset = 4, .len = 6 }, fl.alloc(6).?);
    try testing.expectEqual(null, fl.alloc(1));
}

test "free merges neighbors back into one range" {
    var fl: FreeList = try .init(testing.allocator, 12);
    defer fl.deinit(testing.allocator);
    const a = fl.alloc(4).?;
    const b = fl.alloc(4).?;
    const c = fl.alloc(4).?;
    try fl.free(testing.allocator, a);
    try fl.free(testing.allocator, c);
    try testing.expectEqual(2, fl.free_ranges.items.len);
    try fl.free(testing.allocator, b); // merges with both sides
    try testing.expectEqual(1, fl.free_ranges.items.len);
    try testing.expectEqual(Range{ .offset = 0, .len = 12 }, fl.free_ranges.items[0]);
}

test "first fit reuses a hole" {
    var fl: FreeList = try .init(testing.allocator, 12);
    defer fl.deinit(testing.allocator);
    const a = fl.alloc(4).?;
    _ = fl.alloc(4).?;
    try fl.free(testing.allocator, a);
    try testing.expectEqual(Range{ .offset = 0, .len = 3 }, fl.alloc(3).?);
    try testing.expectEqual(5, fl.freeCount());
}
