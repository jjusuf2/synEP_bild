import numpy as np
import noctiluca as nl

def get_trajs(dt, condition):
    all_trajs = np.load(f'data/{dt}s_{condition}.npy')
    trajs = []
    for track_idx in range(len(all_trajs)):
        traj = all_trajs[track_idx,:,:]
        row_all_nan = np.all(np.isnan(traj), 1)
        non_nan_rows = np.where(~row_all_nan)[0]
        endpoint = non_nan_rows[-1] + 1 if len(non_nan_rows) > 0 else 0
        traj = traj[:endpoint]
        traj = traj / 1000  # convert to µm
        trajs.append(traj)
    return trajs

def downsample_trajs(trajs, sample_every):
    trajs_downsampled = []
    for i in range(len(trajs)):
        traj = trajs[i]
        trajs_downsampled.append(traj[::sample_every,:])
    return trajs_downsampled

def get_original_downsampled_trajs(condition, min_traj_len=50, print_output=True):

    original_trajs = get_trajs(30, condition)
    original_trajs = [traj for traj in original_trajs if len(traj)>min_traj_len]
    original_trajs_hrs = np.sum([len(traj) for traj in original_trajs])*30/60/60

    downsampled_trajs = downsample_trajs(get_trajs(5, condition), 6)
    downsampled_trajs = [traj for traj in downsampled_trajs if len(traj)>min_traj_len]
    downsampled_trajs_hours = np.sum([len(traj) for traj in downsampled_trajs])*30/60/60

    if print_output:
        print(f'Condition name: {condition}')
        print(f'Loaded {len(original_trajs)} trajectories with ∆t=30s')
        print(f'  ↳ filtered to {len(original_trajs)} trajectories with ≥{min_traj_len} frames')
        print(f'    ({original_trajs_hrs:.0f} hours total)')
        print(f'Loaded {len(downsampled_trajs)} trajectories with ∆t=5s and downsampled')
        print(f'  ↳ filtered to {len(downsampled_trajs)} trajectories with ≥{min_traj_len} frames')
        print(f'    ({downsampled_trajs_hours:.0f} hours total)')
        print()

    # make noctiluca TaggedSets

    original_trajs_nl = nl.TaggedSet()
    for traj_arr in original_trajs:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 30
        original_trajs_nl.add(traj)

    downsampled_trajs_nl = nl.TaggedSet()
    for traj_arr in downsampled_trajs:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 30
        downsampled_trajs_nl.add(traj)

    return original_trajs_nl, downsampled_trajs_nl

def get_30s_5s_trajs(condition, min_traj_len=50, print_output=True):

    trajs_30s = get_trajs(30, condition)
    trajs_30s = [traj for traj in trajs_30s if len(traj)>min_traj_len]
    trajs_30s_hrs = np.sum([len(traj) for traj in trajs_30s])*30/60/60

    trajs_5s = get_trajs(5, condition)
    trajs_5s = [traj for traj in trajs_5s if len(traj)>min_traj_len]
    trajs_5s_hrs = np.sum([len(traj) for traj in trajs_5s])*30/60/60

    if print_output:
        print(f'Condition name: {condition}')
        print(f'Loaded {len(trajs_30s)} trajectories with ∆t=30s')
        print(f'  ↳ filtered to {len(trajs_30s)} trajectories with ≥{min_traj_len} frames')
        print(f'    ({trajs_30s_hrs:.0f} hours total)')
        print(f'Loaded {len(trajs_5s)} trajectories with ∆t=5s and downsampled')
        print(f'  ↳ filtered to {len(trajs_5s)} trajectories with ≥{min_traj_len} frames')
        print(f'    ({trajs_5s_hrs:.0f} hours total)')
        print()

    # make noctiluca TaggedSets

    trajs_30s_nl = nl.TaggedSet()
    for traj_arr in trajs_30s:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 30
        trajs_30s_nl.add(traj)

    trajs_5s_nl = nl.TaggedSet()
    for traj_arr in trajs_5s:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 5
        trajs_5s_nl.add(traj)

    return trajs_30s_nl, trajs_5s_nl