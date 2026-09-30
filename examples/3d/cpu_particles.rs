//! CPU-simulated particles drawn through a [`GpuBatchedMesh3d`].
use bevy::{
    camera::{primitives::Aabb, Hdr},
    math::{Affine3, Affine3Ext, Vec3A},
    pbr::gpu_instance_batch::{
        GpuBatchedMesh3d, GpuInstanceBatchPlugin, GpuInstanceBatchReservations, GpuMeshInstance,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        renderer::RenderQueue, sync_world::MainEntityHashMap, Extract, ExtractSchedule, Render,
        RenderApp, RenderSystems,
    },
};
use chacha20::ChaCha8Rng;
use rand::{RngExt, SeedableRng};

const MAX_PARTICLES: u32 = 8192;
const SPAWN_RATE: f32 = 2400.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.8, 0.0);
const RESTITUTION: f32 = 0.6;
const EMITTER_POSITION: Vec3 = Vec3::new(0.0, 0.5, 0.0);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GpuInstanceBatchPlugin)
        .add_plugins(CpuParticlesRenderPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                move_obstacles,
                simulate_particles,
                write_particle_instances,
                update_obstacle_glow,
            )
                .chain(),
        )
        .run();
}

struct Particle {
    position: Vec3,
    velocity: Vec3,
    age: f32,
    lifetime: f32,
}

#[derive(Component)]
struct CpuParticles {
    particles: Vec<Particle>,
    instances: Vec<GpuMeshInstance>,
    rng: ChaCha8Rng,
    spawn_accumulator: f32,
    next_slot: usize,
}

impl CpuParticles {
    fn new(capacity: u32) -> Self {
        Self {
            particles: (0..capacity)
                .map(|_| Particle {
                    position: Vec3::ZERO,
                    velocity: Vec3::ZERO,
                    age: 0.0,
                    lifetime: 0.0,
                })
                .collect(),
            instances: vec![GpuMeshInstance::default(); capacity as usize],
            rng: ChaCha8Rng::seed_from_u64(19878367467712),
            spawn_accumulator: 0.0,
            next_slot: 0,
        }
    }

    fn spawn(&mut self) {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.particles.len();
        let angle = self.rng.random_range(0.0..std::f32::consts::TAU);
        let spread = self.rng.random_range(0.0..2.0);
        self.particles[slot] = Particle {
            position: EMITTER_POSITION,
            velocity: Vec3::new(
                angle.cos() * spread,
                self.rng.random_range(9.0..12.0),
                angle.sin() * spread,
            ),
            age: 0.0,
            lifetime: self.rng.random_range(3.0..5.0),
        };
    }
}

#[derive(Component)]
struct Obstacle {
    radius: f32,
    orbit_radius: f32,
    orbit_speed: f32,
    phase: f32,
    glow: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(32.0, 32.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.12),
            perceptual_roughness: 0.4,
            metallic: 0.2,
            ..default()
        })),
    ));

    let sphere = meshes.add(Sphere::new(1.0).mesh().ico(4).unwrap());
    for (i, (radius, orbit_radius, orbit_speed)) in
        [(1.0, 2.5, 0.7), (0.7, 1.5, -1.1), (0.8, 3.5, 0.4)]
            .into_iter()
            .enumerate()
    {
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.18, 0.22, 0.28),
                perceptual_roughness: 0.35,
                metallic: 0.6,
                ..default()
            })),
            Transform::from_scale(Vec3::splat(radius)),
            Obstacle {
                radius,
                orbit_radius,
                orbit_speed,
                phase: i as f32 * 2.1,
                glow: 0.0,
            },
        ));
    }

    commands.spawn((
        GpuBatchedMesh3d {
            mesh: meshes.add(Cuboid::new(0.05, 0.05, 0.05)),
            max_capacity: MAX_PARTICLES,
        },
        CpuParticles::new(MAX_PARTICLES),
        Aabb {
            center: Vec3A::new(0.0, 4.0, 0.0),
            half_extents: Vec3A::new(16.0, 8.0, 16.0),
        },
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.8, 0.5),
            emissive: LinearRgba::rgb(4.0, 2.0, 0.6),
            ..default()
        })),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Hdr,
        Bloom::default(),
        Transform::from_xyz(0.0, 6.0, 18.0).looking_at(Vec3::new(0.0, 3.5, 0.0), Vec3::Y),
    ));
}

fn move_obstacles(time: Res<Time>, mut obstacles: Query<(&mut Transform, &Obstacle)>) {
    for (mut transform, obstacle) in &mut obstacles {
        let angle = obstacle.phase + time.elapsed_secs() * obstacle.orbit_speed;
        transform.translation = Vec3::new(
            angle.cos() * obstacle.orbit_radius,
            2.0 + obstacle.radius + (angle * 1.3).sin(),
            angle.sin() * obstacle.orbit_radius,
        );
    }
}

fn simulate_particles(
    time: Res<Time>,
    mut emitter: Single<&mut CpuParticles>,
    mut obstacles: Query<(&Transform, &mut Obstacle)>,
) {
    let dt = time.delta_secs().min(1.0 / 30.0);

    emitter.spawn_accumulator += SPAWN_RATE * dt;
    while emitter.spawn_accumulator >= 1.0 {
        emitter.spawn_accumulator -= 1.0;
        emitter.spawn();
    }

    for particle in &mut emitter.particles {
        if particle.age >= particle.lifetime {
            continue;
        }
        particle.age += dt;
        particle.velocity += GRAVITY * dt;
        particle.position += particle.velocity * dt;

        if particle.position.y < 0.0 {
            particle.position.y = 0.0;
            particle.velocity.y = -particle.velocity.y * RESTITUTION;
        }

        for (transform, mut obstacle) in &mut obstacles {
            let offset = particle.position - transform.translation;
            let distance_squared = offset.length_squared();
            if distance_squared >= obstacle.radius * obstacle.radius {
                continue;
            }
            let normal = offset.normalize_or(Vec3::Y);
            particle.position = transform.translation + normal * obstacle.radius;
            let into_surface = particle.velocity.dot(normal);
            if into_surface < 0.0 {
                particle.velocity -= (1.0 + RESTITUTION) * into_surface * normal;
            }
            obstacle.glow += 0.02;
        }
    }
}

fn write_particle_instances(mut emitter: Single<&mut CpuParticles>) {
    let CpuParticles {
        particles,
        instances,
        ..
    } = &mut **emitter;
    for (index, (particle, instance)) in particles.iter().zip(instances).enumerate() {
        if particle.age >= particle.lifetime {
            *instance = GpuMeshInstance::default();
            continue;
        }
        let speed = particle.velocity.length();
        let size = 1.0 - particle.age / particle.lifetime;
        let rotation = Quat::from_rotation_arc(Vec3::Y, particle.velocity / speed.max(1e-4));
        let world_from_local = Affine3::from_scale_rotation_translation(
            Vec3::new(size, size * (1.0 + speed * 0.3), size),
            rotation,
            particle.position,
        );
        *instance = GpuMeshInstance {
            world_from_local: world_from_local.to_transpose(),
            is_active: 1,
            tag: index as u32,
            pad: [0; 2],
        };
    }
}

fn update_obstacle_glow(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut obstacles: Query<(&mut Obstacle, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (mut obstacle, material) in &mut obstacles {
        obstacle.glow *= (-3.0 * time.delta_secs()).exp();
        if let Some(mut material) = materials.get_mut(material) {
            let glow = obstacle.glow.min(20.0);
            material.emissive = LinearRgba::rgb(glow * 0.6, glow * 0.25, glow * 0.05);
        }
    }
}

struct CpuParticlesRenderPlugin;

impl Plugin for CpuParticlesRenderPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedParticleInstances>()
            .add_systems(ExtractSchedule, extract_particle_instances)
            .add_systems(
                Render,
                upload_particle_instances.in_set(RenderSystems::PrepareResources),
            );
    }
}

#[derive(Resource, Default)]
struct ExtractedParticleInstances(MainEntityHashMap<Vec<GpuMeshInstance>>);

fn extract_particle_instances(
    mut extracted: ResMut<ExtractedParticleInstances>,
    emitters: Extract<Query<(Entity, &CpuParticles)>>,
) {
    extracted
        .0
        .retain(|entity, _| emitters.contains(entity.id()));
    for (entity, particles) in &emitters {
        let instances = extracted.0.entry(entity.into()).or_default();
        instances.clear();
        instances.extend_from_slice(&particles.instances);
    }
}

fn upload_particle_instances(
    extracted: Res<ExtractedParticleInstances>,
    reservations: Res<GpuInstanceBatchReservations>,
    queue: Res<RenderQueue>,
) {
    for (&entity, instances) in &extracted.0 {
        let Some(reservation) = reservations.get(entity) else {
            continue;
        };
        let count = instances.len().min(reservation.max_capacity() as usize);
        queue.write_buffer(
            reservation.output_buffer(),
            0,
            bytemuck::cast_slice(&instances[..count]),
        );
    }
}
