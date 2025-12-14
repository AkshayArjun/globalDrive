## 1 Design Choices and Algorithms

### A. Path Smoothing: Quintic Hermite Spline Generation
To enable smooth and safe motion for a differential drive robot, the path smoothing algorithm must ensure continuity not just in Position ($C^0$) and Velocity ($C^1$), but also in Acceleration ($C^2$). Standard cubic splines or linear interpolation often result in discontinuous acceleration profiles, causing jerky motion and torque spikes in the motors.

To address this, I implemented a **Quintic (5th-order) Hermite Spline Generator** (`FastSplineGenerator`).

#### Algorithm Detail: Quintic Hermite Spline Formulation
Unlike standard B-Splines, a Hermite spline is defined explicitly by the physical boundary conditions at the endpoints of each segment, rather than by external control points.

**Mathematical Formulation**
For a segment between two waypoints $P_0$ and $P_1$, the position $p(u)$ is defined by a 5th-order polynomial:
$$
p(u) = c_0 + c_1 u + c_2 u^2 + c_3 u^3 + c_4 u^4 + c_5 u^5
$$
where $u \in [0, 1]$ is the normalized time parameter.

To find the coefficient vector $C = [c_0, ..., c_5]^T$, we define a geometry vector $G$ containing the boundary constraints:
$$
G = [p_0, v_0, a_0, p_1, v_1, a_1]^T
$$
* **$p_{0,1}$**: Position at start/end (Waypoint coordinates).
* **$v_{0,1}$**: Velocity (Tangent vector) at start/end.
* **$a_{0,1}$**: Acceleration (Curvature) at start/end.

The coefficients are solved via the linear mapping $C = M \cdot G$, where $M$ is the **Hermite Basis Matrix**. This matrix was pre-calculated and hardcoded into `FastSplineGenerator` to avoid runtime inversion:

$$
M = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.5 & 0 & 0 & 0 \\
-10 & -6 & -1.5 & 10 & -4 & 0.5 \\
15 & 8 & 1.5 & -15 & 7 & -1 \\
-6 & -3 & -0.5 & 6 & -3 & 0.5
\end{bmatrix}
$$

**Design Rationale:**
1.  **$C^2$ Continuity:** The Quintic Hermite formulation explicitly matches acceleration $a_0$ of segment $i$ with $a_1$ of segment $i-1$, ensuring the robot never experiences instantaneous torque spikes.
2.  **Explicit Tangent Control:** By explicitly calculating tangents based on the geometry of three consecutive waypoints, we can adjust the "corner sharpness" heuristic to prevent the robot from swinging wide on sharp turns.

### B. Trajectory Generation (Time-Parameterization)
The assignment required a time-parameterized trajectory `(x, y, t)`.
* **Logic:** The system calculates the cumulative distance $s$ along the generated spline.
* **Time Allocation:** Timestamps are generated using a constant average speed profile: $t_i = s_i / v_{avg}$.
* **Output:** The final trajectory matrix includes a dedicated Time column, allowing for strict temporal analysis if required.

### C. Controller Architecture: Model Predictive Control (MPC)
I chose a linearized **MPC controller** over a standard PID.
* **Reasoning:** PID controllers struggle with the coupled dynamics of differential drive robots. MPC solves a convex optimization problem at every time step ($10Hz$), optimizing a finite horizon ($N=40$) to minimize the following cost function:

  $$J = \sum_{k=0}^{N-1} \left( \mathbf{e}_k^T \mathbf{Q} \mathbf{e}_k + \mathbf{u}_k^T \mathbf{R} \mathbf{u}_k \right)$$

  Where:
  * $\mathbf{e}_k = [x_{ref} - x, y_{ref} - y, \theta_{ref} - \theta]^T$ is the state error vector.
  * $\mathbf{u}_k = [v, \omega]^T$ is the control input vector.
  * **$\mathbf{Q}$ (State Cost Matrix):** A diagonal matrix that penalizes tracking error. High values in $\mathbf{Q}$ force the robot to track the path aggressively (prioritizing accuracy).
  * **$\mathbf{R}$ (Input Cost Matrix):** A diagonal matrix that penalizes control effort. High values in $\mathbf{R}$ force the robot to accelerate and turn gently (prioritizing smoothness and energy efficiency).

* **Adaptability:** The MPC naturally handles constraints (e.g., $v_{max}$, $\omega_{max}$) and anticipates curves, significantly reducing overshoot compared to reactive controllers.
---

## 2 Architectural Decisions

### Separation of Concerns (Modularity)
Instead of a monolithic script, the system is architected into three distinct modules:
1.  **Planner (`FastSplineGenerator`):** Handles pure geometry and math. It is unaware of ROS or the robot's hardware.
2.  **Controller (`MPCController`):** A pure math class that solves optimization problems. It is decoupled from the robot's communication layer.
3.  **Manager (`NavigationNode`):** A ROS2 node that acts as the "glue," managing sensor inputs (`/odom`), executing the planner once, and looping the controller at $10Hz$.
* **Rationale:** This modularity ensures testability. The planner can be swapped for A* (Nav2) or the controller swapped for PID without breaking the rest of the system.

### Cascaded Control Hierarchy
The system uses a two-layered control approach:
* **Global Layer (Offline):** The Spline Generator plans the full trajectory once at startup. This handles the long-term goal.
* **Local Layer (Online):** The MPC re-plans a short trajectory ($N=40$ steps) continuously at $10Hz$.
* **Rationale:** Global planning is computationally expensive and cannot react to immediate state changes. Local MPC is fast and handles immediate disturbances, providing the agility required for dynamic navigation.


---

## 3 Extension to Real Robots

To deploy this on a physical TurtleBot3 or custom platform:
1.  **State Estimation:** The current system relies on perfect Odometry. On a real robot, I would implement an **Extended Kalman Filter (EKF)** fusing Wheel Odometry with IMU data (and potentially LiDAR SLAM like AMCL) to provide the robust `[x, y, theta, v, w]` estimate required by the MPC.
2.  **Computation Management:** The convex optimization (MPC) is computationally heavy. For embedded hardware (e.g., Raspberry Pi 4), I would reduce the MPC Horizon $N$ from 50 to 20 or switch to a lighter solver like OSQP to maintain the $10Hz$ control loop.
3.  **Map Integration:** Instead of hardcoded waypoints (`mission.csv`), I would integrate a Global Planner (like Nav2's A*) to generate the initial waypoints dynamically based on a static map.

---

## 4  Obstacle Avoidance

I tried implemented a reactive safety layer using **Control Barrier Functions (C3BF)**, which acts as a "Safety Filter" between the MPC and the Motors.

* **Algorithm:**
    1.  **LiDAR Clustering:** Raw laser scans are grouped into clusters to detect nearest obstacles.
    2.  **Safety Bubble:** A virtual radius is calculated around the obstacle (Physical Radius + Safety Margin + 0.35m).
    3.  **QP Solver:** A secondary Quadratic Program checks if the MPC's desired velocity command $u_{mpc}$ will cause a collision within the next $t$ seconds. If unsafe, it finds the closest safe velocity $u_{safe}$ that satisfies the barrier condition: $\dot{h}(x) \geq -\gamma h(x)$.

* **Deadlock Resolution:**
    A common failure mode in reactive avoidance is "getting stuck" (MPC pushes forward, Safety pushes back):
    Im yet to fix this issue and completely implement c3bf into this project

---

## 5 AI Tools Used

* **LLMs like gemini** were used to understand and research through different algorithmns 


* it was used to help quickly prototype ideas to code. However debugging and correction of logical errors were still done manually. 

