# Objective

Let user compute shaders author mesh instances that the standard mesh pipeline draws: culled per view, drawn with any `Material`, with shadows, prepass, and deferred, as one indirect draw per batch. This is the low-level primitive that GPU particle systems need to draw lit meshes (see #20569), and the piece bevy_hanabi's lighting request (djeedai/bevy_hanabi#215) would build on.

Today every `MeshInputUniform` slot is owned by one entity and written from ECS extraction. The only way to draw compute-written instances is a custom render command with a hand-rolled instance buffer (`custom_shader_instancing`), which gets no material, culling, or shadow support.

# Solution

A `GpuBatchedMesh3d { mesh, max_capacity }` entity, paired with a `MeshMaterial3d`, reserves a persistent output buffer of `max_capacity` `GpuMeshInstance { world_from_local, is_active, tag }` slots. A producer writes them from compute in the `GpuInstanceBatchSystems::Publish` set of the `RenderGraph` schedule, recording through `RenderContext`. A renderer-owned materialize pass then copies the slots into a GPU-only tail of the mesh preprocessing input and culling buffers, applying a CPU-owned template for mesh, material, and flags. A range-unpacking pass expands the batch into ordinary preprocessing work items, so frustum and occlusion culling, indirect parameter building, and drawing are unchanged; inactive slots are skipped there.

Design points:

- Producers never see `MeshInputUniform`. The template is rebuilt every frame from the mesh allocator and material bindings, so mesh and material changes need no re-reservation.
- The output buffer persists across frames, and device recovery replaces it. Newly allocated slots are inactive and draw nothing.
- Sorted phases treat a batch as one item, sorted by the entity's `Aabb` center. Instances inside a batch are not sorted, and compaction does not preserve buffer order.
- `GetFullBatchData::get_instance_batch` is the only new trait surface; it defaults to `None`, so `Mesh2d` and third-party batchers are unaffected.

Changes outside the new module that reviewers should look at:

- `Mesh3dVisibility` is now the visibility class of 3D mesh draws, required by `Mesh3d` and by `GpuBatchedMesh3d`, instead of `Mesh3d` itself. Class lookups in material, prepass, light, wireframe, and mesh collection moved to it, and the material instance sweep keys on it too.
- `MeshCullingData` gains a per-slot `is_active: u32` in its trailing padding, read by `mesh_preprocess` under frustum culling. Stride is unchanged.
- Merged sorted phase items now have an empty `batch_range`, and `SortedRenderPhase::render_range` draws every item with a non-empty range, advancing one item at a time, instead of skipping `batch_range.len()` items after a draw. `batch_range.len()` used to mean both "instances drawn" and "phase items consumed", which an instance batch breaks. It also fixes a bug on main: the transmission node renders its phase in item-index steps, and a step starting on a merged follower drew that follower again with its stale range, doubling it on the CPU batching path or drawing it at slot 0's transform on the GPU path.
- Retained transparent items keep the sort center they were queued with (#24826). Batches refresh theirs in place each frame in `PhaseSort`. When #25708 lands, its refresh can take over by reading `RenderMeshDraws`, and the batch-specific system goes away.

Requires GPU culling support; the plugin warns once and draws nothing otherwise. Not supported: motion vectors, transmissive materials, per-instance sorting, 2D.

# Testing

`gpu_particles` example (new): a compute-simulated swarm drawn with `StandardMaterial`, casting and receiving shadows. Space pauses the simulation; the last published state keeps rendering.

Ran `gpu_particles`, `transparency_3d`, `mesh2d`, and `mesh2d_alpha_mode` on macOS/Metal (M2 Max) with GPU preprocessing fully supported, no validation errors. The transmission double-draw was reproduced against a device on main with a recording draw function and confirmed fixed on this branch. Not tested on Vulkan, DX12, or WebGPU, or on adapters without culling support.

# Showcase

TODO screenshot of `gpu_particles`.

## Migration Guides

- `Mesh3dVisibility` is the visibility class of 3D meshes.
- Merged sorted phase items have an empty `batch_range`.
