import numpy as np
import matplotlib.pyplot as plt

import ssms


def plot_traj_and_drift(out, show=False):
    traj = out["metadata"]["trajectory"]
    drift = out["metadata"]["drift"]
    max_t = out["metadata"]["max_t"]
    time_traj = np.linspace(0, max_t, num=len(traj))
    time_drift = np.linspace(0, max_t, num=drift.shape[0])
    for i in range(traj.shape[1]):
        traj[:, i][traj[:, i] == -999] = out["choices"][0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(time_traj, traj, linewidth=3)
    axs[0].set_title("Trajectory")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Trajectory")

    axs[1].plot(time_drift, drift, linewidth=3)
    axs[1].set_title("Drift")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Drift")

    plt.tight_layout()
    if show:
        plt.show()
    return plt


if __name__ == "__main__":

    delta_t = 0.001
    max_t = 3
    samples = 1
    theta_base = {
        "a": 1,
        "z" : 0.5,
        "t" : 0,
        "tcoh" : 1,
        "dcoh" : -1,
        "tonset" : 0.8,
        "donset" : 0.2,
        "toffset": 1.0,
        "doffset": 0.4}

    cfg = ssms.config.get_model_config()
    cfg.keys()

    out_leak = ssms.basic_simulators.simulator.simulator(
        model="ds_conflict_stimflexons_leak_drift",
        theta=theta_base | {"tinit": 0.01, "tfixedp": 0.01, "tslope": 0,
                            "dinit": 0, "dslope": 0, "g": 0.5},
        n_samples=samples, delta_t=delta_t, max_t=max_t,
        no_noise = True
    )
    out_leak2 = ssms.basic_simulators.simulator.simulator(
        model="conflict_stimflex_leak2_drift",
        theta=theta_base | {"v_t": 0.01, "v_d": 0.005, "g_t": 0, "g_d": 0.5},
        n_samples=samples, delta_t=delta_t, max_t=max_t,
        no_noise = True
    )
    out_leak2_noise = ssms.basic_simulators.simulator.simulator(
        model="conflict_stimflex_leak2_drift",
        theta=theta_base | {"v_t": 5, "v_d": 0.1, "g_t": 0, "g_d": 1},
        n_samples=samples, delta_t=delta_t, max_t=max_t,
        no_noise = False
    )    
    np.mean(out_leak2_noise["choices"] == 1) ## mean accuracy
    plot_traj_and_drift(out_leak, show = True)
    plot_traj_and_drift(out_leak2, show = True)
    plot_traj_and_drift(out_leak2_noise, show = True)
