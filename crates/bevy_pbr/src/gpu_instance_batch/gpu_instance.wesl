// One instance slot published by a GPU producer. Zeroed slots are inactive.
struct GpuMeshInstance {
    world_from_local: mat3x4<f32>,
    is_active: u32,
    tag: u32,
    skin_index: u32,
    flags: u32,
}

const GPU_MESH_INSTANCE_SKIN: u32 = 1u;
