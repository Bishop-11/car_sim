# car_sim

A lightweight 2D kinematic driving simulator, built as the testbed for learning
classical (PID) and RL-based lane-following control. No game engine, no 3D
assets — just numpy/OpenCV rendering a procedurally generated road and a
simulated front camera, all wired up over ROS2 topics so classical and RL
controllers can be dropped in as separate packages later.

## Pipeline

```
                    ┌────────────────────────────────────────┐
                    │               sim_node                  │
                    │                                           │
 /car/control ─────▶│  RoadGenerator ─▶ CarState ─▶ renderer   │─────▶ /car/camera/image_raw  (front camera, for lane detection)
 (String:            │  (procedural       (kinematic   (pinhole  │─────▶ /car/chase/image_raw   (3rd-person game view)
  left/right/         │   curvature)        model)       camera +  │─────▶ /car/game_over          (off-road / victory flag)
  straight)           │                                  map view) │
                    └────────────────────────────────────────┘
                                      ▲
                                      │ /car/control
                    ┌─────────────────┴─────────────────┐
                    │        manual_control_node          │   (manual play only)
                    │  pygame window: shows /car/chase,    │
                    │  arrow keys → /car/control            │
                    └───────────────────────────────────┘
```

`sim_node` is the single source of truth: it owns the car's pose, the
procedurally generated road, and the camera model, and renders every frame.
`manual_control_node` is a thin I/O adapter — it only runs during manual play,
displaying the chase view and turning arrow-key state into `/car/control`
messages. An automatic controller (PID, RL, ...) is just another node that
subscribes to `/car/camera/image_raw` and publishes `/car/control` — no
changes to `sim_node` needed.

## Example output

The two published views, mid-drive on a curve (rendered with the default
params in `config/sim_params.yaml`):

| `/car/camera/image_raw` (front camera, for lane detection) | `/car/chase/image_raw` (game view) |
|---|---|
| ![camera feed](docs/camera_view.png) | ![chase/game view](docs/chase_view.png) |

## Nodes

| Node | Runs when | Does |
|---|---|---|
| `sim_node` | always | kinematics, road generation, rendering, game rules |
| `manual_control_node` | manual play only | pygame chase-view window + arrow-key → `/car/control` |

## Topics & services

| Name | Type | Direction (sim_node) | Notes |
|---|---|---|---|
| `/car/control` | `std_msgs/String` | sub | `"left"` / `"right"` / `"straight"`, published continuously |
| `/car/camera/image_raw` | `sensor_msgs/Image` | pub | front-facing pinhole camera feed, BGR8 |
| `/car/chase/image_raw` | `sensor_msgs/Image` | pub | third-person "game view" — used as the manual-play window and available for visualization in automatic-control mode |
| `/car/game_over` | `std_msgs/Bool` | pub | true once off-road or victory; episode is frozen (see below) |
| `/car/reset` | `std_srvs/Trigger` | service | respawns the car and regenerates the road (same seed) |

## How the road & car work

- **Road**: `RoadGenerator` builds an endless centerline ahead of the car via
  a rate-limited random walk on curvature (a new target curvature is sampled
  every `segment_length` meters, and the current curvature is steered toward
  it at most `curvature_rate` per meter, so turns are smooth). A rolling
  window (`lookahead`/`keep_behind`) is kept generated around the car and
  pruned behind it. The first `straight_start_m` meters are forced straight
  as a warm-up. Road width is constant; the finish line is a fixed point
  `DIFFICULTY * 1000` meters down the road (see below).
- **Car**: `CarState` is a point-mass kinematic model — constant forward
  speed, commanded yaw rate while turning, no wheel/tire dynamics.
- **Camera**: `PinholeCamera` is a real pinhole projection (intrinsics from
  resolution + HFOV, extrinsics from the car's pose + mount height/pitch),
  not a pseudo-3D approximation — world-space road/lane geometry is
  projected into the image with proper near-plane clipping, so the feed is
  geometrically consistent for a lane-detection pipeline to consume.
- **Game rules**: off-road (lateral offset beyond `road_width/2`) → game
  over; reaching the finish line → victory (checkered pattern drawn on the
  road as you approach it). Either way, `sim_node` freezes — no further
  kinematics, road generation, or per-tick rendering — and just republishes
  a single cached frame with a "GAME OVER"/"VICTORY!" banner until
  `/car/reset` is called.

## Parameters (`config/sim_params.yaml`)

| Group | Params |
|---|---|
| Car | `car_speed` (m/s), `yaw_rate` (rad/s) |
| Road | `road_seed`, `road_width`, `waypoint_spacing`, `segment_length`, `max_curvature`, `curvature_rate`, `lookahead`, `keep_behind`, `straight_start_m` |
| Sim | `tick_rate` (Hz) |
| Camera | `camera_width`, `camera_height`, `camera_hfov_deg`, `camera_mount_height`, `camera_pitch_deg`, `camera_render_distance` |
| Chase/game view | `chase_width_px`, `chase_height_px`, `chase_scale` (px/m), `chase_car_x_frac`, `chase_car_y_frac` (car's position as a *fraction* of the canvas, so the whole layout scales seamlessly if you resize the canvas) |

`road_seed` is also a launch argument (`road_seed:=<int>`) for reproducible
episodes, e.g. when comparing PID vs. RL runs or varying it across RL
training episodes.

Difficulty (`DIFFICULTY` in `sim_node.py`, finish line = `DIFFICULTY * 1000`
meters) is a code-level constant for now, not a launch parameter.

## Running it

Manual play (sim + pygame control window):
```
ros2 launch car_sim manual_play.launch.py [road_seed:=123]
```

Sim only (for an automatic controller package to launch alongside):
```
ros2 launch car_sim sim_only.launch.py [road_seed:=123]
```

## Repo layout

This package is one repo in a multi-repo ROS2 workspace
(`lane_drive_ws/src/`) — `car_sim` here handles the game/simulation only.
Classical and RL controllers live in their own separate repos
(`car_pid_control`, `car_rl_control`), each subscribing to
`/car/camera/image_raw` and publishing `/car/control`, so they're
interchangeable without touching this package.
