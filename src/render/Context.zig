//! Vulkan 1.4 instance, surface, physical device and logical device.
const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("zglfw");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.vulkan);

const Context = @This();

const validation_layer = "VK_LAYER_KHRONOS_validation";

vkb: vk.BaseWrapper,
vki: *vk.InstanceWrapper,
vkd: *vk.DeviceWrapper,
instance: vk.InstanceProxy,
debug_messenger: vk.DebugUtilsMessengerEXT,
surface: vk.SurfaceKHR,
pdev: vk.PhysicalDevice,
props: vk.PhysicalDeviceProperties,
mem_props: vk.PhysicalDeviceMemoryProperties,
device: vk.DeviceProxy,
queue_family: u32,
queue: vk.QueueProxy,

pub fn init(gpa: Allocator, window: *glfw.Window) !Context {
    var self: Context = undefined;
    self.vkb = vk.BaseWrapper.load(glfw.getInstanceProcAddress);

    // Instance: GLFW's surface extensions + debug utils; validation in Debug if installed.
    const glfw_exts = try glfw.getRequiredInstanceExtensions();
    var exts: std.ArrayList([*:0]const u8) = .empty;
    defer exts.deinit(gpa);
    try exts.appendSlice(gpa, glfw_exts);
    const debug = builtin.mode == .Debug and try hasLayer(gpa, self.vkb, validation_layer);
    if (debug) try exts.append(gpa, vk.extensions.ext_debug_utils.name);
    const layers: []const [*:0]const u8 = if (debug) &.{validation_layer} else &.{};
    log.info("validation layers: {s}", .{if (debug) "on" else "off"});

    const debug_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
        .message_severity = .{ .warning_bit_ext = true, .error_bit_ext = true },
        .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
        .pfn_user_callback = debugCallback,
    };
    // Wrappers are allocated before the handles they load, so that no failure
    // can leave a live instance or device without the means to destroy it.
    self.vki = try gpa.create(vk.InstanceWrapper);
    errdefer gpa.destroy(self.vki);
    self.vkd = try gpa.create(vk.DeviceWrapper);
    errdefer gpa.destroy(self.vkd);

    const instance = try self.vkb.createInstance(&.{
        .p_next = if (debug) &debug_info else null,
        .p_application_info = &.{
            .p_application_name = "ft_vox",
            .application_version = 0,
            .p_engine_name = "ft_vox",
            .engine_version = 0,
            .api_version = @bitCast(vk.API_VERSION_1_4),
        },
        .enabled_layer_count = @intCast(layers.len),
        .pp_enabled_layer_names = layers.ptr,
        .enabled_extension_count = @intCast(exts.items.len),
        .pp_enabled_extension_names = exts.items.ptr,
    }, null);

    self.vki.* = vk.InstanceWrapper.load(instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);
    self.instance = vk.InstanceProxy.init(instance, self.vki);
    errdefer self.instance.destroyInstance(null);

    self.debug_messenger = if (debug) try self.instance.createDebugUtilsMessengerEXT(&debug_info, null) else .null_handle;
    errdefer if (debug) self.instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);

    try glfw.createWindowSurface(instance, window, null, &self.surface);
    errdefer self.instance.destroySurfaceKHR(self.surface, null);

    try self.pickPhysicalDevice(gpa);
    self.props = self.instance.getPhysicalDeviceProperties(self.pdev);
    self.mem_props = self.instance.getPhysicalDeviceMemoryProperties(self.pdev);
    log.info("GPU: {s}", .{std.mem.sliceTo(&self.props.device_name, 0)});

    // Device: one graphics+present queue, the features the renderer relies on.
    var f14: vk.PhysicalDeviceVulkan14Features = .{ .push_descriptor = .true, .maintenance_5 = .true };
    var f13: vk.PhysicalDeviceVulkan13Features = .{ .p_next = &f14, .dynamic_rendering = .true, .synchronization_2 = .true, .maintenance_4 = .true };
    var f12: vk.PhysicalDeviceVulkan12Features = .{ .p_next = &f13, .buffer_device_address = .true, .draw_indirect_count = .true, .timeline_semaphore = .true, .scalar_block_layout = .true };
    const f10: vk.PhysicalDeviceFeatures2 = .{ .p_next = &f12, .features = .{ .shader_int_64 = .true, .multi_draw_indirect = .true, .draw_indirect_first_instance = .true, .depth_clamp = .true } };
    const priority = [_]f32{1};
    const dev = try self.instance.createDevice(self.pdev, &.{
        .p_next = &f10,
        .queue_create_info_count = 1,
        .p_queue_create_infos = &.{.{ .queue_family_index = self.queue_family, .queue_count = 1, .p_queue_priorities = &priority }},
        .enabled_extension_count = 1,
        .pp_enabled_extension_names = &.{vk.extensions.khr_swapchain.name},
    }, null);
    self.vkd.* = vk.DeviceWrapper.load(dev, self.vki.dispatch.vkGetDeviceProcAddr.?);
    self.device = vk.DeviceProxy.init(dev, self.vkd);
    self.queue = vk.QueueProxy.init(self.device.getDeviceQueue(self.queue_family, 0), self.vkd);
    return self;
}

pub fn deinit(self: *Context, gpa: Allocator) void {
    self.device.destroyDevice(null);
    self.instance.destroySurfaceKHR(self.surface, null);
    if (self.debug_messenger != .null_handle) self.instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);
    self.instance.destroyInstance(null);
    gpa.destroy(self.vkd);
    gpa.destroy(self.vki);
    self.* = undefined;
}

/// First device exposing Vulkan 1.4, the required features and a graphics+present queue.
fn pickPhysicalDevice(self: *Context, gpa: Allocator) !void {
    const pdevs = try self.instance.enumeratePhysicalDevicesAlloc(gpa);
    defer gpa.free(pdevs);
    for (pdevs) |pdev| {
        const props = self.instance.getPhysicalDeviceProperties(pdev);
        const name = std.mem.sliceTo(&props.device_name, 0);
        if (@as(vk.Version, @bitCast(props.api_version)).minor < 4) {
            log.info("skipping {s}: Vulkan 1.4 required", .{name});
            continue;
        }
        if (missingFeature(self.instance, pdev)) |missing| {
            log.info("skipping {s}: missing feature {s}", .{ name, missing });
            continue;
        }
        const families = try self.instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, gpa);
        defer gpa.free(families);
        for (families, 0..) |fam, i| {
            const idx: u32 = @intCast(i);
            if (!fam.queue_flags.graphics_bit or !fam.queue_flags.compute_bit) continue;
            if (try self.instance.getPhysicalDeviceSurfaceSupportKHR(pdev, idx, self.surface) != .true) continue;
            self.pdev = pdev;
            self.queue_family = idx;
            return;
        }
    }
    return error.NoSuitableGpu;
}

fn missingFeature(instance: vk.InstanceProxy, pdev: vk.PhysicalDevice) ?[]const u8 {
    var f14: vk.PhysicalDeviceVulkan14Features = .{};
    var f13: vk.PhysicalDeviceVulkan13Features = .{ .p_next = &f14 };
    var f12: vk.PhysicalDeviceVulkan12Features = .{ .p_next = &f13 };
    var f10: vk.PhysicalDeviceFeatures2 = .{ .p_next = &f12, .features = .{} };
    instance.getPhysicalDeviceFeatures2(pdev, &f10);
    const checks = .{
        .{ f14.push_descriptor, "pushDescriptor" },
        .{ f14.maintenance_5, "maintenance5" },
        .{ f13.dynamic_rendering, "dynamicRendering" },
        .{ f13.synchronization_2, "synchronization2" },
        .{ f13.maintenance_4, "maintenance4" },
        .{ f12.buffer_device_address, "bufferDeviceAddress" },
        .{ f12.draw_indirect_count, "drawIndirectCount" },
        .{ f12.timeline_semaphore, "timelineSemaphore" },
        .{ f12.scalar_block_layout, "scalarBlockLayout" },
        .{ f10.features.shader_int_64, "shaderInt64" },
        .{ f10.features.multi_draw_indirect, "multiDrawIndirect" },
        .{ f10.features.draw_indirect_first_instance, "drawIndirectFirstInstance" },
        .{ f10.features.depth_clamp, "depthClamp" },
    };
    inline for (checks) |c| if (c[0] != .true) return c[1];
    return null;
}

fn hasLayer(gpa: Allocator, vkb: vk.BaseWrapper, name: []const u8) !bool {
    const layers = try vkb.enumerateInstanceLayerPropertiesAlloc(gpa);
    defer gpa.free(layers);
    for (layers) |l| if (std.mem.eql(u8, std.mem.sliceTo(&l.layer_name, 0), name)) return true;
    return false;
}

fn debugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    _: vk.DebugUtilsMessageTypeFlagsEXT,
    data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    const msg = if (data) |d| d.p_message orelse "?" else "?";
    if (severity.error_bit_ext) log.err("{s}", .{msg}) else log.warn("{s}", .{msg});
    return .false;
}

/// Index of a memory type allowed by `type_bits` with all `flags`.
pub fn findMemoryType(self: *const Context, type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
    for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |t, i| {
        if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and t.property_flags.contains(flags)) return @intCast(i);
    }
    return error.NoSuitableMemoryType;
}
