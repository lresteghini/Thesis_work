# This script aims to replicate the results in Killoran's article. This code implements the system with the harmonic oscillators

using LinearAlgebra
using DifferentialEquations
using Plots
using DelimitedFiles
using SciMLBase

# we first define the two constants we need to have consistent units of measurement with c=h=1 and energies in cm^-1

const kb_unit_converter = 0.6950348
const cm_inv_to_eV_converter = 8065.544

# we also define some useful functions

commutator(a, b) = a*b - b*a
anticommutator(a,b) = a*b + b*a

limb_form(Operator, rho) = (Operator * rho * adjoint(Operator)) - 0.5 * anticommutator(adjoint(Operator) * Operator, rho)

# we construct a vector to hold the values of Γ_load over which we'll run our simulations

n_loads = 375
load_step = 0.0333

load = zeros(n_loads)

b = 300

# the following loop fills the vector. It's a very artisanal process as of now

for i in 1:n_loads
    load[i] = b * exp((i-1) * load_step) - b
end

# we also initialize the vectors that we'll later fill with the measurements

final_currents = zeros(n_loads)

final_voltages = zeros(n_loads)

final_powers = zeros(n_loads)

# here, we construct the electronic Hamiltonian

n_qbit = 4

J = 100

qbit_basis = [zeros(ComplexF64, n_qbit) for _ in 1:n_qbit]

for i in 1:n_qbit
    qbit_basis[i][i] = 1.0
end

E_i = ComplexF64[-10000, 300, 300, 0]

J_ij = [0 0 0 0
        0 0 J 0
        0 J 0 0
        0 0 0 0]

H_electronic =  zeros(ComplexF64, n_qbit, n_qbit)

for i in 1:n_qbit
    H_electronic[i, i] = E_i[i]
end

H_electronic = H_electronic + J_ij          

# here, we construct the harmonic oscillator Hamiltonian
# D is the truncature of the ho levels

D = 6

omega = 200

ho_basis = [zeros(ComplexF64, D) for _ in 1:D]

for i in 1:D
    ho_basis[i][i] = 1.0
end

a = zeros(ComplexF64, D, D)

for i in 1:D-1
    a[i, i+1] = sqrt(i)
end

a_dagger = adjoint(a)

H_ho = omega * a_dagger * a

# here, we construct the interaction Hamiltonian

g2 = 55
g3 = 55 

I_ho = Matrix(I, D, D)

I_qbit = Matrix(I, n_qbit, n_qbit)

H_interaction = kron(qbit_basis[2] * adjoint(qbit_basis[2]), g2 * (a + a_dagger), I_ho) + 
                kron(qbit_basis[3] * adjoint(qbit_basis[3]), I_ho, g3 * (a + a_dagger))

# and finally, the total Hamiltonian

H_tot = kron(H_electronic, I_ho, I_ho) +
        kron(I_qbit, H_ho, I_ho) +
        kron(I_qbit, I_ho, H_ho) +
        H_interaction

# H_tot is a matrix. We define the Hamiltonian as a function of time for the DifferentialEquations solver

Hamiltonian(t) = H_tot

# here, we define the eigenvalues and eigenvectors of the electronic Hamiltonian
# eigen_basis[1] and [4] are the same as the positional basis
# while [2] and [3] are the coupled eigenstates of the second and third qbit

eigen_basis = copy(qbit_basis)

eigen_basis[2] = (qbit_basis[2] + qbit_basis[3])/sqrt(2)  
eigen_basis[3] = (qbit_basis[2] - qbit_basis[3])/sqrt(2)

eigen_energy = copy(E_i)

eigen_energy[2] = adjoint(eigen_basis[2]) * H_electronic * eigen_basis[2]
eigen_energy[3] = adjoint(eigen_basis[3]) * H_electronic * eigen_basis[3]

# here we construct the Lindblad hot transition. The parameters for the coefficient are taken from Killoran's article

Transition_hot = kron(eigen_basis[2] * adjoint(qbit_basis[1]), I_ho, I_ho)

Transition_reverse_hot = kron(qbit_basis[1] * adjoint(eigen_basis[2]), I_ho, I_ho)

Coeff_hot = 0.01 * 60000

Coeff_reverse_hot = 0.01 * (60000 + 1)

Limb_hot(rho) = Coeff_hot * limb_form(Transition_hot, rho) +
                Coeff_reverse_hot * limb_form(Transition_reverse_hot, rho)

# here we construct the Lindblad operator for the harmonic oscillator thermal bath

Eccitation_ho2 = kron(I_qbit, a_dagger, I_ho)

Damping_ho2 = kron(I_qbit, a, I_ho)

Eccitation_ho3 = kron(I_qbit, I_ho, a_dagger)

Damping_ho3 = kron(I_qbit, I_ho, a)

Temp_ho = 293

Population_ho = 1 / (exp(omega/(kb_unit_converter * Temp_ho)) - 1)

Limb_ho(rho) = 5.3 * (
            Population_ho * (limb_form(Eccitation_ho2, rho) + limb_form(Eccitation_ho3, rho)) +
            (Population_ho + 1) * (limb_form(Damping_ho2, rho) + limb_form(Damping_ho3, rho))
)

# here, we construc the Lindblad cold transitions. The parameters are still taken from Killoran's article

Gamma_cold = 8.07
Temp_cold = 293

function limb_term(population, op, rev_op, rho)
    return population * limb_form(rev_op, rho) +
        (1 + population) * limb_form(op, rho)
end

populations = Float64[]
for_trans = Matrix{ComplexF64}[]
back_trans = Matrix{ComplexF64}[]

# here we calculate the populations and the transition operators

for i in 1:n_qbit
    for j in 1:n_qbit
        if i != j
            energy_diff = abs(eigen_energy[i] - eigen_energy[j])
            if energy_diff > 1e-9
                ij_population = 1 / (exp(energy_diff / (kb_unit_converter * Temp_cold)) - 1)
            else
                continue
            end

            Transition_forward = kron(eigen_basis[j] * adjoint(eigen_basis[i]), I_ho, I_ho)
            Transition_backward = kron(eigen_basis[i] * adjoint(eigen_basis[j]), I_ho, I_ho)

            
            push!(populations, ij_population)
            push!(for_trans, Transition_forward)
            push!(back_trans, Transition_backward)
        end
    end
end

# the Lindblad term is just the sum of all the transition operators and the populations

Limb_cold(rho) = Gamma_cold * sum(limb_term(populations[k], for_trans[k], back_trans[k], rho) for k in 1:length(populations))

# here, we construct the Lindblad load term, which is a function of Γ_load 

function Limb_load(rho, Gamma_load)
    Transition_load = kron(qbit_basis[1] * adjoint(qbit_basis[4]), I_ho, I_ho)
    return Gamma_load * limb_form(Transition_load, rho)
end

# here, we sum all the Lindblad terms

function Limb_tot(rho, Gamma_load)
    return Limb_hot(rho) + Limb_ho(rho) + Limb_load(rho, Gamma_load) + Limb_cold(rho)
end

# here, we define the time derivative of ρ. The function also has the parameter p, which is used to change Γ_load between simulations

function rho_primo(rho, p, t)
    Gamma_load = p[1]
    H_t = Hamiltonian(t)
    return 1im * commutator(rho, H_t) + Limb_tot(rho, Gamma_load)
end

# here, we define the equilibrium condition at which to stop the simulations
# we check whether both the first and last qbits' populations have changed less than a defined threshold in the last step
#and if yes, we stop the simulation

P_q1_full = kron(qbit_basis[1] * adjoint(qbit_basis[1]), I_ho, I_ho)
P_q4_full = kron(qbit_basis[4] * adjoint(qbit_basis[4]), I_ho, I_ho)

function steady_state_condition(rho, t, integrator)
    drho = get_du(integrator)

    pop_q1_rate = abs(real(tr(drho * P_q1_full)))
    pop_q4_rate = abs(real(tr(drho * P_q4_full)))

    return max(pop_q1_rate, pop_q4_rate) < 1e-1
end

affect!(integrator) = terminate!(integrator)

cb = DiscreteCallback(steady_state_condition, affect!)

# here, we construct the starting ρ for the first simulation
# for the following ones, we'll just start from the final ρ of the previous run

ho_2_state = 2

rho0_ho_2 = ho_basis[ho_2_state] * adjoint(ho_basis[ho_2_state])

ho_3_state = 2

rho0_ho_3 = ho_basis[ho_3_state] * adjoint(ho_basis[ho_3_state])

excited_qbit = 1

rho0_electronic = qbit_basis[excited_qbit] * adjoint(qbit_basis[excited_qbit])

rho0 = kron(rho0_electronic, rho0_ho_2, rho0_ho_3)

# here, we open the files in which we'll write the results

csv_header_gv = ["Gamma_load", "Voltage_V"]
open("latest/gamma_voltage_relationship.csv", "w") do io
    writedlm(io, reshape(csv_header_gv, 1, :), ',')
end

csv_header_vip = ["Voltage", "Current", "Power"]
open("latest/VIP_data.csv", "w") do io
    writedlm(io, reshape(csv_header_vip, 1, :), ',')
end

#start_time = time_ns() for convenience, you can print to console the time it takes for the simulations to complete

# this is the main simulation loop. We load the Γ_load, perform the simulation until the stop condition is met, and then calculate
# currents and voltages, and write them to file

for index in 1:n_loads

    current_gamma = load[index]

    p = (current_gamma,)

    tspan = (0.0, 2.0)
    prob = ODEProblem(rho_primo, rho0, tspan, p)
    sol = solve(prob, Tsit5(), callback = cb, save_everystep = false, reltol = 1e-8, abstol = 1e-8)
    rhof = sol[2]

    pop_q1 = real(tr(rhof * P_q1_full))
    pop_q4 = real(tr(rhof * P_q4_full))

    final_currents[index] = real(pop_q4 * current_gamma)/0.01  #0.01 = gammmaH as in Killoran's article

    Eg = E_i[4] - E_i[1]

    final_voltages[index] = real(Eg + Temp_cold * kb_unit_converter * log(pop_q4 / pop_q1))/ cm_inv_to_eV_converter

    current_power = final_currents[index] * final_voltages[index]
    final_powers[index] = current_power
    
    open("latest/gamma_voltage_relationship.csv", "a") do io
        new_data_gv = [load[index] final_voltages[index]]
        writedlm(io, new_data_gv, ',')
    end

    open("latest/VIP_data.csv", "a") do io
        new_data_vip = [final_voltages[index] final_currents[index] current_power]
        writedlm(io, new_data_vip, ',')
    end

    #elapsed_time = (time_ns() - start_time) / 1e9

    #print("$(index): gamma $(current_gamma) voltage $(final_voltages[index]) (Elapsed: $(round(elapsed_time, digits=2)) s)\n")
    
    global rho0 = rhof
    
end

# we then plot our results, and save them

plot_IV = plot(final_voltages, final_currents,
                 label="I-V Curve",
                 xlabel="Voltage (V)",
                 ylabel="Current (I)",
                 legend=:bottomleft)
title!(plot_IV, "I-V Characteristic")
plot!(twinx(),  final_voltages, final_powers,
                 label="Power Curve",
                 ylabel="Power (P)",
                 legend=:topright,
                 color=:red)
title!(plot_IV, "Power vs. Voltage")
display(plot_IV)

savefig(plot_IV, "latest/iv_power_plot.png")