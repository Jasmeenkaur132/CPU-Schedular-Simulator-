from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#inputs

def get_input():
    """
    Prompts the user to enter process details interactively.
    Returns:
        processes (list): list of [PID, arrival_time, burst_time]
        quantum   (int) : time quantum for Round Robin
    """
    print("\n╔══════════════════════════════════════╗")
    print("║   CPU Scheduling Simulator v1.0      ║")
    print("║   COSC-519 | Jasmeen Kaur            ║")
    print("╚══════════════════════════════════════╝\n")

    n = int(input("Enter number of processes: "))
    processes = []

    for i in range(n):
        pid     = f"P{i+1}"
        arrival = int(input(f"  {pid} - Arrival Time: "))
        burst   = int(input(f"  {pid} - Burst Time:   "))
        processes.append([pid, arrival, burst])

    quantum = int(input("\nEnter Time Quantum for Round Robin: "))
    return processes, quantum


#fcfs

def fcfs(processes):
    """
    Non-preemptive. Processes are served in order of arrival time.
    If the CPU is idle (no process has arrived yet), time jumps
    forward to the next arrival — recorded as an idle block.

    Args:
        processes: list of [PID, arrival_time, burst_time]
    Returns:
        results: list of [PID, AT, BT, CT, TAT, WT]
        gantt  : list of (PID, start, end) for Gantt chart
    """
    # Sort by arrival time
    procs = sorted(processes, key=lambda x: x[1])

    time    = 0      # current CPU clock
    results = []
    gantt   = []     # stores (pid, start_time, end_time)

    for pid, arrival, burst in procs:
        # CPU is idle — no process has arrived yet
        if time < arrival:
            gantt.append(("idle", time, arrival))
            time = arrival

        start      = time
        time      += burst          # process runs to completion
        completion = time
        tat        = completion - arrival
        wt         = tat - burst

        gantt.append((pid, start, completion))
        results.append([pid, arrival, burst, completion, tat, wt])

    return results, gantt

#sjf algo

def sjf(processes):
    """
    Non-preemptive. At each scheduling point, selects the arrived
    process with the smallest burst time. Once started, the
    process runs to completion.

    Args:
        processes: list of [PID, arrival_time, burst_time]
    Returns:
        results: list of [PID, AT, BT, CT, TAT, WT]
        gantt  : list of (PID, start, end) for Gantt chart
    """
    procs = sorted(processes, key=lambda x: x[1])  # sort by arrival
    n     = len(procs)
    done  = [False] * n    # tracks completed processes
    time  = 0
    results, gantt = [], []

    for _ in range(n):
        # Find all processes that have arrived and are not done
        available = [
            (i, p) for i, p in enumerate(procs)
            if not done[i] and p[1] <= time
        ]

        # CPU is idle — jump to next arriving process
        if not available:
            next_arrival = min(p[1] for i, p in enumerate(procs) if not done[i])
            gantt.append(("idle", time, next_arrival))
            time = next_arrival
            available = [(i, p) for i, p in enumerate(procs) if not done[i] and p[1] <= time]

        # Pick process with shortest burst time (greedy selection)
        idx, (pid, arrival, burst) = min(available, key=lambda x: x[1][2])

        start      = time
        time      += burst
        completion = time
        tat        = completion - arrival
        wt         = tat - burst

        done[idx] = True
        gantt.append((pid, start, completion))
        results.append([pid, arrival, burst, completion, tat, wt])

    return results, gantt


# ─────────────────────────────────────────────────────────────
#  ALGORITHM 3: SRTF — Shortest Remaining Time First
# ─────────────────────────────────────────────────────────────

def srtf(processes):
    """
    Preemptive version of SJF. At every clock tick, the process
    with the shortest REMAINING burst time is executed.
    A new arrival can preempt the current process if it has
    a shorter remaining time — this is recorded in the Gantt chart.

    Args:
        processes: list of [PID, arrival_time, burst_time]
    Returns:
        results: list of [PID, AT, BT, CT, TAT, WT]
        gantt  : list of [PID, start, end] for Gantt chart
    """
    procs     = sorted(processes, key=lambda x: x[1])
    n         = len(procs)
    remaining = [p[2] for p in procs]   # remaining burst times
    completion= [0] * n
    time      = 0
    done      = 0               # number of finished processes
    last_pid  = None            # tracks context switches
    gantt     = []

    while done < n:
        # Find processes that have arrived and still need CPU
        available = [
            i for i in range(n)
            if procs[i][1] <= time and remaining[i] > 0
        ]

        # CPU idle — no process available yet
        if not available:
            if last_pid != "idle":
                gantt.append(["idle", time, time + 1])
            else:
                gantt[-1][2] += 1
            time     += 1
            last_pid  = "idle"
            continue

        # Select process with minimum remaining time
        idx = min(available, key=lambda i: remaining[i])
        pid = procs[idx][0]

        # Merge consecutive ticks for same process in Gantt
        if pid != last_pid:
            gantt.append([pid, time, time + 1])
        else:
            gantt[-1][2] += 1      # extend current block

        # Execute one tick
        remaining[idx] -= 1
        time           += 1
        last_pid        = pid

        # Check if process finished
        if remaining[idx] == 0:
            completion[idx] = time
            done           += 1

    # Build results from completion times
    results = []
    for i in range(n):
        pid, arrival, burst = procs[i]
        tat = completion[i] - arrival
        wt  = tat - burst
        results.append([pid, arrival, burst, completion[i], tat, wt])

    return results, gantt

#round robin algo
def round_robin(processes, quantum):
    """
    Preemptive. Each process gets up to `quantum` time units.
    If not finished, it returns to the back of the ready queue.
    Designed for fairness — no process can monopolize the CPU.

    Args:
        processes: list of [PID, arrival_time, burst_time]
        quantum  : maximum time slice per turn
    Returns:
        results: list of [PID, AT, BT, CT, TAT, WT]
        gantt  : list of (PID, start, end) for Gantt chart
    """
    procs     = sorted(processes, key=lambda x: x[1])
    n         = len(procs)
    remaining = [p[2] for p in procs]    # remaining burst per process
    completion= [0] * n
    time      = 0
    queue     = deque()          # ready queue (circular)
    visited   = [False] * n
    gantt     = []
    ptr       = 0                # pointer to track new arrivals

    # Add first process to queue
    queue.append(0)
    visited[0] = True

    while queue:
        idx = queue.popleft()
        pid, arrival, burst = procs[idx]

        # Run for min(quantum, remaining) time units
        run_time = min(quantum, remaining[idx])

        gantt.append((pid, time, time + run_time))
        time          += run_time
        remaining[idx]-= run_time

        # Add any newly arrived processes to the queue
        while ptr + 1 < n and procs[ptr + 1][1] <= time:
            ptr += 1
            if not visited[ptr]:
                queue.append(ptr)
                visited[ptr] = True

        # If process finished, record completion time
        if remaining[idx] == 0:
            completion[idx] = time
        else:
            # Not finished — re-add to back of queue
            queue.append(idx)

    # Build results
    results = []
    for i in range(n):
        pid, arrival, burst = procs[i]
        tat = completion[i] - arrival
        wt  = tat - burst
        results.append([pid, arrival, burst, completion[i], tat, wt])

    return results, gantt

#result table

def print_results(results, algo_name):
    """
    Prints a formatted table of scheduling results and averages.

    Args:
        results   : output from any scheduling algorithm
        algo_name : name of the algorithm (for heading)
    Returns:
        avg_tat (float), avg_wt (float)
    """
    print(f"\n--- {algo_name} ---")
    print(f"{'PID':<6} {'Arrival':<10} {'Burst':<8} {'Completion':<13} {'Turnaround':<13} {'Waiting'}")
    print("-" * 62)

    for r in results:
        print(f"{r[0]:<6} {r[1]:<10} {r[2]:<8} {r[3]:<13} {r[4]:<13} {r[5]}")

    avg_tat = sum(r[4] for r in results) / len(results)
    avg_wt  = sum(r[5] for r in results) / len(results)

    print(f"\n  Avg Turnaround Time : {avg_tat:.2f}")
    print(f"  Avg Waiting Time    : {avg_wt:.2f}")

    return avg_tat, avg_wt


#Gantt Charts
COLORS = [
    "#2563eb", "#059669", "#d97706", "#9333ea",
    "#db2777", "#0891b2", "#65a30d", "#ea580c"
]

def draw_gantt(gantt, title, pid_list):
    """
    Draws a horizontal Gantt chart using matplotlib and saves it as PNG.
    Idle blocks are shown in light gray with diagonal stripes.

    Args:
        gantt   : list of (pid, start, end)
        title   : chart title (also used as filename)
        pid_list: ordered list of process IDs for color mapping
    """
    # Map each PID to a color
    color_map = {pid: COLORS[i % len(COLORS)] for i, pid in enumerate(pid_list)}
    color_map["idle"] = "#e8e6df"   # gray for idle

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#f5f4f0')
    ax.set_facecolor('#f5f4f0')

    for (pid, start, end) in gantt:
        color = color_map.get(pid, "#aaa")
        ax.barh(0, end - start, left=start, height=0.5,
                color=color, edgecolor="white", linewidth=0.8)

        # Label the block if wide enough
        if end - start >= 1:
            label = pid if pid != "idle" else "idle"
            ax.text((start + end) / 2, 0, label,
                    ha="center", va="center",
                    color="white" if pid != "idle" else "#9e9b92",
                    fontweight="bold", fontsize=9)

    ax.set_xlim(0, max(e for _, _, e in gantt) + 1)
    ax.set_yticks([])
    ax.set_xlabel("Time Units", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1e3a5f")

    # Build legend (exclude idle)
    legend = [
        mpatches.Patch(color=color_map[p], label=p)
        for p in pid_list if p in color_map
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    plt.tight_layout()
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved → {fname}")
    plt.show()

#Comparison Charts

def draw_comparison(stats):
    """
    Draws a grouped bar chart comparing Avg Waiting Time and
    Avg Turnaround Time across all 4 scheduling algorithms.

    Args:
        stats: list of (algo_name, avg_wt, avg_tat)
    """
    algos   = [s[0] for s in stats]
    avg_wt  = [s[1] for s in stats]
    avg_tat = [s[2] for s in stats]

    x     = range(len(algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#f5f4f0')
    ax.set_facecolor('#f5f4f0')

    bars1 = ax.bar([i - width/2 for i in x], avg_wt,  width,
                   label="Avg Waiting Time",    color="#2563eb", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], avg_tat, width,
                   label="Avg Turnaround Time", color="#059669", alpha=0.85)

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(algos)
    ax.set_ylabel("Time Units")
    ax.set_title("Algorithm Performance Comparison", fontsize=13,
                 fontweight="bold", color="#1e3a5f")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("Comparison_Graph.png", dpi=150, bbox_inches='tight')
    print("  Saved → Comparison_Graph.png")
    plt.show()


if __name__ == "__main__":
    # Step 1: Get process input from user
    processes, quantum = get_input()
    pid_list = [p[0] for p in processes]
    stats    = []   # collect (name, avg_wt, avg_tat) for comparison

    print("\n════════════════ RESULTS ════════════════")

    # Step 2: Run FCFS
    r, g = fcfs(processes)
    tat, wt = print_results(r, "FCFS")
    draw_gantt(g, "FCFS Gantt Chart", pid_list)
    stats.append(("FCFS", wt, tat))

    # Step 3: Run SJF
    r, g = sjf(processes)
    tat, wt = print_results(r, "SJF")
    draw_gantt(g, "SJF Gantt Chart", pid_list)
    stats.append(("SJF", wt, tat))

    # Step 4: Run SRTF
    r, g = srtf(processes)
    tat, wt = print_results(r, "SRTF")
    draw_gantt(g, "SRTF Gantt Chart", pid_list)
    stats.append(("SRTF", wt, tat))

    # Step 5: Run Round Robin
    r, g = round_robin(processes, quantum)
    tat, wt = print_results(r, f"Round Robin (q={quantum})")
    draw_gantt(g, f"Round Robin Gantt Chart (q={quantum})", pid_list)
    stats.append((f"RR(q={quantum})", wt, tat))

    # Step 6: Show comparison
    print("\n════════════════ COMPARISON ════════════════")
    draw_comparison(stats)

    # Step 7: Print verdict
    best = min(stats, key=lambda x: x[1])
    print(f"\n✅ Best algorithm for this workload: {best[0]}")
    print(f"   Lowest avg waiting time: {best[1]:.2f}")
    print("\n✅ All done! Check your folder for saved PNG charts.")
