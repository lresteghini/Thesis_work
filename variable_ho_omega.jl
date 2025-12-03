using LinearAlgebra
using DifferentialEquations
using Plots
using DelimitedFiles
using SciMLBase # per terminate!
#using GR

# lo script per i conti senza oscillatori è identico, 
# ma con tutte le parti di codice riguardanti gli oscillatori rimosse 
# (ed i prodotti tensore ed il loro bagno termico)

const kb_unit_converter = 0.6950348
const cm_inv_to_eV_converter = 8065.544

commutator(a, b) = a*b - b*a
anticommutator(a,b) = a*b + b*a

limb_form(Operator, rho) = (Operator * rho * adjoint(Operator)) - 0.5 * anticommutator(adjoint(Operator) * Operator, rho)

n_omega = 26
omega_step = 10

omegas = zeros(n_omega)

b = 100 #200 - (n_omega*omega_step/2)

for i in 1:23
    omegas[i] = b + omega_step * (i-1) 
end

omegas[24] = 275
omegas[25] = 279
omegas[26] = 200 + 55*sqrt(2)

# costruisce un vettore per i valori gamma_load
# esponenziale fatto abbastanza ad occhio sulla base dei grafici
# di relazione tra gamma e V

final_currents = zeros(n_omega)

final_voltages = zeros(n_omega)

# inizializza vettori per le misure

# da qui costruisco base dei qbit
# e Hamiltoniana elettronica

n_qbit = 4

J = 100

qbit_basis = [zeros(ComplexF64, n_qbit) for _ in 1:n_qbit]            #Base nel sottospazio di singole eccitazioni

for i in 1:n_qbit
    qbit_basis[i][i] = 1.0
end         

E_i = ComplexF64[-10000, 300, 300, 0]

J_ij = [0 0 0 0
        0 0 J 0
        0 J 0 0
        0 0 0 0]                                                         #Coupling matrix editabile a mano

H_electronic =  zeros(ComplexF64, n_qbit, n_qbit)

for i in 1:n_qbit
    H_electronic[i, i] = E_i[i]
end

H_electronic = H_electronic + J_ij          

# Fine parte elettronica

D = 6                                                  #Troncamento dell'oscillatore armonico

I_ho = Matrix(I, D, D)                                 #Per comodità

I_qbit = Matrix(I, n_qbit, n_qbit)

#omega = 200                                            #Frequenza dell'oscillatore armonico

ho_basis = [zeros(ComplexF64, D) for _ in 1:D]         #Base dell'oscillatore legato al qbit 2

for i in 1:D
    ho_basis[i][i] = 1.0
end

#ho_3_basis = ho_2_basis                               #La base ha la stessa forma

a = zeros(ComplexF64, D, D)

# operatore distruzione 

for i in 1:D-1
    a[i, i+1] = sqrt(i)
end

a_dagger = adjoint(a)                                  #Operatori creazione e distruzione degli oscillatori armonici

################################################################

#H_ho(omega) = omega * a_dagger * a                            #Hamiltoniana dell'oscillatore armonico

#g2 = 55
#g3 = 55                                                 #Coupling tra oscillatori e qbit

#H_interaction = kron(qbit_basis[2] * adjoint(qbit_basis[2]), g2 * (a + a_dagger), I_ho) + 
#                kron(qbit_basis[3] * adjoint(qbit_basis[3]), I_ho, g3 * (a + a_dagger))

#H_tot = kron(H_electronic, I_ho, I_ho) +
#        kron(I_qbit, H_ho, I_ho) +
#        kron(I_qbit, I_ho, H_ho) +
#        H_interaction

#Hamiltonian(t) = H_tot

###############################################################


    ## Definisco le transizioni da 1 a 2

epsilon_1 = (qbit_basis[2] + qbit_basis[3])/sqrt(2)  # Speriamo bene

Transition_hot = kron(epsilon_1 * adjoint(qbit_basis[1]), I_ho, I_ho)

Transition_reverse_hot = kron(qbit_basis[1] * adjoint(epsilon_1), I_ho, I_ho)

Coeff_hot = 0.01 * 60000                                    # Presi dall'articolo (0.01 * 60 000)

Coeff_reverse_hot = 0.01 * (60000 + 1)

Limb_hot(rho) = Coeff_hot * limb_form(Transition_hot, rho) +
                Coeff_reverse_hot * limb_form(Transition_reverse_hot, rho)


## Definisco il bagno termico degli oscillatori armonici

Eccitation_ho2 = kron(I_qbit, a_dagger, I_ho)

Damping_ho2 = kron(I_qbit, a, I_ho)

Eccitation_ho3 = kron(I_qbit, I_ho, a_dagger)

Damping_ho3 = kron(I_qbit, I_ho, a)

Temp_ho = 293

#Population_ho = 1 / (exp(omega/(kb_unit_converter * Temp_ho)) - 1)

#Limb_ho(rho) = 5.3 * (
#            Population_ho * (limb_form(Eccitation_ho2, rho) + limb_form(Eccitation_ho3, rho)) +
#            (Population_ho + 1) * (limb_form(Damping_ho2, rho) + limb_form(Damping_ho3, rho))
#)

## Definisco il reservoir freddo 

Gamma_cold = 8.07
Temp_cold = 293

# parametri presi dall'articolo 

function limb_term(population, op, rev_op, rho)
    return population * limb_form(rev_op, rho) +
        (1 + population) * limb_form(op, rho)
end


populations = Float64[]
for_trans = Matrix{ComplexF64}[]
back_trans = Matrix{ComplexF64}[]

eigen_basis = copy(qbit_basis)

eigen_basis[2] = (qbit_basis[2] + qbit_basis[3])/sqrt(2)  
eigen_basis[3] = (qbit_basis[2] - qbit_basis[3])/sqrt(2)

eigen_energy = copy(E_i)

eigen_energy[2] = adjoint(eigen_basis[2]) * H_electronic * eigen_basis[2]
eigen_energy[3] = adjoint(eigen_basis[3]) * H_electronic * eigen_basis[3]

ordered_eigen_energy = copy(eigen_energy)

ordered_eigen_energy[1] = eigen_energy[2]
ordered_eigen_energy[2] = eigen_energy[3]
ordered_eigen_energy[3] = eigen_energy[4]
ordered_eigen_energy[4] = eigen_energy[1]

ordered_eigen_basis = copy(eigen_basis)

ordered_eigen_basis[1] = eigen_basis[2]
ordered_eigen_basis[2] = eigen_basis[3]
ordered_eigen_basis[3] = eigen_basis[4]
ordered_eigen_basis[4] = eigen_basis[1]

# here we calculate the populations and the transition operators

for i in 1:n_qbit
    for j in (i + 1):n_qbit
        if i != j
            energy_diff = abs(ordered_eigen_energy[i] - ordered_eigen_energy[j])
            if energy_diff > 1e-9
                ij_population = 1 / (exp(energy_diff / (kb_unit_converter * Temp_cold)) - 1)
            else
                continue
            end

            Transition_forward = kron(ordered_eigen_basis[j] * adjoint(ordered_eigen_basis[i]), I_ho, I_ho)
            Transition_backward = kron(ordered_eigen_basis[i] * adjoint(ordered_eigen_basis[j]), I_ho, I_ho)
            
            push!(populations, ij_population)
            push!(for_trans, Transition_forward)
            push!(back_trans, Transition_backward)
        end
    end
end

# the Lindblad term is just the sum of all the transition operators and the populations

Limb_cold(rho) = Gamma_cold * sum(limb_term(populations[k], for_trans[k], back_trans[k], rho) for k in 1:length(populations))
#forward is the relaxation, backwards is the excitation

start_time = time_ns() # per comodità mia, misuro i tempi di calcolo

function Limb_load(rho, Gamma_load)
    Transition_load = kron(qbit_basis[1] * adjoint(qbit_basis[4]), I_ho, I_ho)
    return Gamma_load * limb_form(Transition_load, rho)
end

# definisco il temp finale

#function Limb_tot(rho, Gamma_load)
    # sommo tutti i termini di Limblad
#    return Limb_hot(rho) + Limb_ho(rho) + Limb_load(rho, Gamma_load) + Limb_cold(rho)
#end

#################################################################################################3

#function rho_primo(rho, p, t)
#    Gamma_load = p[1] # è il valore di gamma_load
#    H_t = Hamiltonian(t)
#    return 1im * commutator(rho, H_t) + Limb_tot(rho, Gamma_load,)
#end

######################################################################################################

# definisco la condizione di stop per la simulazione

P_q1_full = kron(qbit_basis[1] * adjoint(qbit_basis[1]), I_ho, I_ho)
P_q4_full = kron(qbit_basis[4] * adjoint(qbit_basis[4]), I_ho, I_ho)

function steady_state_condition(rho, t, integrator)
    # prendo la derivata di rho
    drho = get_du(integrator)

    # calcolo la variazione delle popolazioni dei qbit
    pop_q1_rate = abs(real(tr(drho * P_q1_full)))
    pop_q4_rate = abs(real(tr(drho * P_q4_full)))

    # controllo se stiano cambiando sufficientemente o no
    return max(pop_q1_rate, pop_q4_rate) < 1e-4
end

# dichiaro che se non cambiano abbastanza la simulazione finisce
affect!(integrator) = terminate!(integrator)

# è il callback
cb = DiscreteCallback(steady_state_condition, affect!)

# ora definisco lo stato iniziale per la prima simulazione
# quelle dopo partiranno dallo stato finale dell'ultimo gamma provato
# per partire da qualcosa che dovrebbe essere vicino abbastanza all'equilibrio 

ho_2_state = 2                                           #Eccitazione iniziale ho

rho0_ho_2 = ho_basis[ho_2_state] * adjoint(ho_basis[ho_2_state])

ho_3_state = 2                                           #Eccitazione iniziale ho

rho0_ho_3 = ho_basis[ho_3_state] * adjoint(ho_basis[ho_3_state])

excited_qbit = 1                                                      #Qbit eccitato a t=0

rho0_electronic = qbit_basis[excited_qbit] * adjoint(qbit_basis[excited_qbit])              #Matrice densità elettronica   

rho0 = kron(rho0_electronic, rho0_ho_2, rho0_ho_3)


# File for Voltage, Current, and Power
csv_header_vip = ["omega", "Voltage", "Current", "Power"]
open("latest_omega_ho/VIP_data.csv", "w") do io
    writedlm(io, reshape(csv_header_vip, 1, :), ',')
end

# Pre-allocate the power array
final_powers = zeros(n_omega)
final_av = zeros(n_omega)

N_op_ho = a_dagger * a
N_op_ho2_full = kron(I_qbit, N_op_ho, I_ho)
N_op_ho3_full = kron(I_qbit, I_ho, N_op_ho)

print("Initialized, starting loop\n")

for index in 1:n_omega

    ho_2_state = 2                                           #Eccitazione iniziale ho

    rho0_ho_2 = ho_basis[ho_2_state] * adjoint(ho_basis[ho_2_state])

    ho_3_state = 2                                           #Eccitazione iniziale ho

    rho0_ho_3 = ho_basis[ho_3_state] * adjoint(ho_basis[ho_3_state])

    excited_qbit = 1                                                      #Qbit eccitato a t=0

    rho0_electronic = qbit_basis[excited_qbit] * adjoint(qbit_basis[excited_qbit])              #Matrice densità elettronica   

    rho0 = kron(rho0_electronic, rho0_ho_2, rho0_ho_3)

    current_omega = omegas[index] #carico gamma dal vettore 

    H_ho = current_omega * a_dagger * a                            #Hamiltoniana dell'oscillatore armonico

    g2 = 55
    g3 = 55                                                 #Coupling tra oscillatori e qbit

    H_interaction = kron(qbit_basis[2] * adjoint(qbit_basis[2]), g2 * (a + a_dagger), I_ho) + 
                kron(qbit_basis[3] * adjoint(qbit_basis[3]), I_ho, g3 * (a + a_dagger))

    H_tot = kron(H_electronic, I_ho, I_ho) +
            kron(I_qbit, H_ho, I_ho) +
            kron(I_qbit, I_ho, H_ho) +
            H_interaction

    Hamiltonian(t) = H_tot

    Population_ho = 1 / (exp(current_omega/(kb_unit_converter * Temp_ho)) - 1)

    Limb_ho(rho) = 5.3 * (
                Population_ho * (limb_form(Eccitation_ho2, rho) + limb_form(Eccitation_ho3, rho)) +
                (Population_ho + 1) * (limb_form(Damping_ho2, rho) + limb_form(Damping_ho3, rho))
    )

    function Limb_tot(rho, Gamma_load)
        # sommo tutti i termini di Limblad
        return Limb_hot(rho) + Limb_ho(rho) + Limb_load(rho, Gamma_load) + Limb_cold(rho)
    end

    function rho_primo(rho, p, t)
        Gamma_load = p[1] # è il valore di gamma_load
        H_t = Hamiltonian(t)
        return 1im * commutator(rho, H_t) + Limb_tot(rho, Gamma_load,)
    end

#    if index == 1
#        rho0 = kron(rho0_electronic, rho0_ho_2, rho0_ho_3)
#    end

#    non reinizializzo rho0 ad ogni run

    
    current_gamma = 5                                                                               #CONSIDER INCREASING IT

    p = (current_gamma)

    tspan = (0.0, 2.0) #tempo massimo di simulazione se non viene fermata prima
    prob = ODEProblem(rho_primo, rho0, tspan, p)
    sol = solve(prob, Tsit5(), callback = cb, save_everystep = false, reltol = 1e-8, abstol = 1e-8)
    rhof = sol[end] #rho finale (il secondo elemento della soluzione, dato che non salvo step intermedi)

    pop_q1 = real(tr(rhof * P_q1_full))
    pop_q4 = real(tr(rhof * P_q4_full))

    final_currents[index] = real(pop_q4 * current_gamma)/0.01  #0.01 = gammmaH

# la corrente mi sembra identica a come definita nei grafici (che sono I/(e*gamma_h))


    Eg = E_i[4] - E_i[1]

    final_voltages[index] = real(Eg + Temp_cold * kb_unit_converter * log(pop_q4 / pop_q1))/ cm_inv_to_eV_converter

# anche questo mi sembra identico a come definito nei grafici e nel paper (e su questo i dati combaciano)

    # Calculate power for the current step
    current_power = final_currents[index] * final_voltages[index]
    final_powers[index] = current_power # Also store it in the main array for the plot

    # --- 2. Append the latest data to files inside the loop ---
    
    # Append to VIP_data.csv
    open("latest_omega_ho/VIP_data.csv", "a") do io
        # Create a 1-row matrix for the new data and append it
        new_data_vip = [current_omega final_voltages[index] final_currents[index] current_power]
        writedlm(io, new_data_vip, ',')
    end

    elapsed_time = (time_ns() - start_time) / 1e9

    print("$(index): temp $(current_omega) voltage $(final_voltages[index]) current $(final_currents[index]) (Elapsed: $(round(elapsed_time, digits=2)) s)\n")

# Giusto per assicurarmi che il programma stia girando senza  problemi
    
    #global rho0 = rhof

# reinizializzo per la run successiva, così che la simulazione parta vicina all'equilibrio 
    
end

#final_powers = final_currents .* final_voltages

# da qui in avanti produco e salvo grafici e salvo dati su file csv 

plot_omegaI = plot(omegas, final_currents,
                 label="omega-I Curve",
                 xlabel="omega",
                 ylabel="Current (I)",
                 legend=:bottomleft)
title!(plot_omegaI, "omega-I Characteristic")
plot!(twinx(),  omegas, final_powers,
                 label="omega-P curve",
                 ylabel="Power (P)",
                 legend=:topright,
                 color=:red)
title!(plot_omegaI, "omega vs. Power")
display(plot_omegaI) # Uncomment to show the plot

savefig(plot_omegaI, "latest_omega_ho/omegaip_plot.png")

#gamma_voltage_data = hcat(temp, final_voltages)


#csv_header = ["Gamma_load", "Voltage_V"]


#open("latest/gamma_voltage_relationship.csv", "w") do io
#    writedlm(io, reshape(csv_header, 1, :), ',')
#    writedlm(io, gamma_voltage_data, ',')
#end

#VIP_data = hcat(final_voltages, final_currents, final_powers)

#csv_header = ["Voltage", "Current", "Power"]


#open("latestVIP_data.csv", "w") do io
#    writedlm(io, reshape(csv_header, 1, :), ',')
#    writedlm(io, VIP_data, ',')
#end