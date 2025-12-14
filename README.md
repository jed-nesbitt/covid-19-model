# 🦠 COVID-19 SEIRS Spread Model (Phase Plane + Phase Volume Exports)

A Python-based **SEIRS** (Susceptible–Exposed–Infectious–Recovered–Susceptible) model for exploring COVID-19 outbreak dynamics using ODE simulation.  
Includes automated exports for:

- 📉 **Phase plane** plots (2D state-space trajectories, e.g., **S vs I**, **E vs I**, **I vs R**)
- 🧊 **Phase volume** plots (3D state-space trajectories, e.g., **S–E–I** or **S–I–R**)
- ⏱️ Standard time-series outputs (S, E, I, R vs time)

---

## Why SEIRS?

SEIRS extends SEIR by allowing **waning immunity** (Recovered → Susceptible), which helps capture **multiple waves** of infection—useful when immunity decays over time or variants reduce protection.

---

## Model Equations

Let \(N = S + E + I + R\).

\[
\frac{dS}{dt} = -\beta \frac{SI}{N} + \omega R
\]
\[
\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E
\]
\[
\frac{dI}{dt} = \sigma E - \gamma I
\]
\[
\frac{dR}{dt} = \gamma I - \omega R
\]

### Parameters
- **β (beta)**: transmission rate  
- **σ (sigma)**: progression rate from exposed → infectious (**1 / incubation period**)  
- **γ (gamma)**: recovery rate (**1 / infectious period**)  
- **ω (omega)**: immunity waning rate (**1 / immunity duration**)  

---

## Features

✅ ODE simulation (SciPy)  
✅ Exports **Population vs Time**
✅ Exports **phase plane** trajectories (2D)  
✅ Exports **phase volume** trajectories (3D)  

---

