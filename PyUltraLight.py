import time
import sys
import numpy as np
import numexpr as ne
import numba
import pyfftw
import h5py
import os
from IPython.core.display import clear_output

hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
light_year = 9.4607e15  # m
solar_mass = 1.989e30  # kg
axion_mass = 1e-22 * 1.783e-36  # kg
G = 6.67e-11  # N m^2 kg^-2
omega_m0 = 0.31
H_0 = 67.7 * 1e3 / (parsec * 1e6)  # s^-1

length_unit = (8 * np.pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** -0.5
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)



####################### FUNCTION TO GENERATE PROGRESS BAR

def prog_bar(iteration_number, progress, tinterval):
    size = 50
    status = ""
    progress = float(progress) / float(iteration_number)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(size * progress))
    text = "\r[{}] {:.0f}% {}{}{}{}".format(
        "-" * block + " " * (size - block), round(progress * 100, 0),
        status, ' The previous step took ', tinterval, ' seconds.')
    sys.stdout.write(text)
    sys.stdout.flush()



####################### FUNCTION TO CONVERT TO DIMENSIONLESS UNITS

def convert(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value / length_unit
        elif (unit == 'km'):
            converted = value * 1e3 / length_unit
        elif (unit == 'pc'):
            converted = value * parsec / length_unit
        elif (unit == 'kpc'):
            converted = value * 1e3 * parsec / length_unit
        elif (unit == 'Mpc'):
            converted = value * 1e6 * parsec / length_unit
        elif (unit == 'ly'):
            converted = value * light_year / length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted = value / mass_unit
        elif (unit == 'solar_masses'):
            converted = value * solar_mass / mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value * solar_mass * 1e6 / mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted = value / time_unit
        elif (unit == 'yr'):
            converted = value * 60 * 60 * 24 * 365 / time_unit
        elif (unit == 'kyr'):
            converted = value * 60 * 60 * 24 * 365 * 1e3 / time_unit
        elif (unit == 'Myr'):
            converted = value * 60 * 60 * 24 * 365 * 1e6 / time_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value * time_unit / length_unit
        elif (unit == 'km/s'):
            converted = value * 1e3 * time_unit / length_unit
        elif (unit == 'km/h'):
            converted = value * 1e3 / (60 * 60) * time_unit / length_unit
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted



####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS

def convert_back(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value * length_unit
        elif (unit == 'km'):
            converted = value / 1e3 * length_unit
        elif (unit == 'pc'):
            converted = value / parsec * length_unit
        elif (unit == 'kpc'):
            converted = value / (1e3 * parsec) * length_unit
        elif (unit == 'Mpc'):
            converted = value / (1e6 * parsec) * length_unit
        elif (unit == 'ly'):
            converted = value / light_year * length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted = value * mass_unit
        elif (unit == 'solar_masses'):
            converted = value / solar_mass * mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value / (solar_mass * 1e6) * mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted = value * time_unit
        elif (unit == 'yr'):
            converted = value / (60 * 60 * 24 * 365) * time_unit
        elif (unit == 'kyr'):
            converted = value / (60 * 60 * 24 * 365 * 1e3) * time_unit
        elif (unit == 'Myr'):
            converted = value / (60 * 60 * 24 * 365 * 1e6) * time_unit
        else:
            raise NameError('Unsupported time unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value / time_unit * length_unit
        elif (unit == 'km/s'):
            converted = value / (1e3) / time_unit * length_unit
        elif (unit == 'km/h'):
            converted = value / (1e3) * (60 * 60) / time_unit * length_unit
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted



########################FUNCTION TO CHECK FOR SOLITON OVERLAP

def overlap_check(candidate, soliton):
    for i in range(len(soliton)):
        m = max(candidate[0], soliton[i][0])
        d_sol = 5.35854 / m
        c_pos = np.array(candidate[1])
        s_pos = np.array(soliton[i][1])
        displacement = c_pos - s_pos
        distance = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2)
        if (distance < 2 * d_sol):
            return False
    return True



############################FUNCTION TO PUT SPHERICAL SOLITON DENSITY PROFILE INTO 3D BOX (Uses pre-computed array)

def initsoliton(funct, xarray, yarray, zarray, position, alpha, f, delta_x):
    for index in np.ndindex(funct.shape):
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = ((xarray[index[0], 0, 0] - position[0]) ** 2 + (yarray[0, index[1], 0] - position[1]) ** 2 + (
        zarray[0, 0, index[2]] - position[2]) ** 2) ** 0.5
        # Utilises soliton profile array out to dimensionless radius 5.6.
        if (np.sqrt(alpha) * distfromcentre <= 5.6):
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]

        else:
            funct[index] = 0
    return funct



######################### FUNCTION TO INITIALIZE SOLITONS AND EVOLVE

def evolve(central_mass, num_threads, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options,
           save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons, start_time):
    print ('Initialising...')


    ##########################################################################################
    #SET INITIAL CONDITIONS

    if (length_units == ''):
        gridlength = length
    else:
        gridlength = convert(length, length_units, 'l')
    if (duration_units == ''):
        t = duration
    else:
        t = convert(duration, duration_units, 't')
    if (duration_units == ''):
        t0 = start_time
    else:
        t0 = convert(start_time, duration_units, 't')
    if (s_mass_unit == ''):
        cmass = central_mass
    else:
        cmass = convert(central_mass, s_mass_unit, 'm')

    Vcell = (gridlength / float(resol)) ** 3

    ne.set_num_threads(num_threads)

    initsoliton_jit = numba.jit(initsoliton)


    ##########################################################################################
    # CREATE THE TIMESTAMPED SAVE DIRECTORY AND CONFIG.TXT FILE

    save_path = os.path.expanduser(save_path)
    tm = time.localtime()

    talt = ['0', '0', '0']
    for i in range(3, 6):
        if tm[i] in range(0, 10):
            talt[i - 3] = '{}{}'.format('0', tm[i])
        else:
            talt[i - 3] = tm[i]
    timestamp = '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(tm[0], '.', tm[1], '.', tm[2], '_', talt[0], ':', talt[1], ':', talt[2], '_', resol)
    file = open('{}{}{}'.format('./', save_path, '/timestamp.txt'), "w+")
    file.write(timestamp)
    os.makedirs('{}{}{}{}'.format('./', save_path, '/', timestamp))
    file = open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+")
    file.write(('{}{}'.format('resol = ', resol)))
    file.write('\n')
    file.write(('{}{}'.format('axion_mass (kg) = ', axion_mass)))
    file.write('\n')
    file.write(('{}{}'.format('length (code units) = ', gridlength)))
    file.write('\n')
    file.write(('{}{}'.format('duration (code units) = ', t)))
    file.write('\n')
    file.write(('{}{}'.format('start_time (code units) = ', t0)))
    file.write('\n')
    file.write(('{}{}'.format('step_factor  = ', step_factor)))
    file.write('\n')
    file.write(('{}{}'.format('central_mass (code units) = ', cmass)))
    file.write('\n\n')
    file.write(('{}'.format('solitons ([mass, [x, y, z], [vx, vy, vz], phase]): \n')))
    for s in range(len(solitons)):
        file.write(('{}{}{}{}{}'.format('soliton', s, ' = ', solitons[s], '\n')))
    file.write(('{}{}{}{}{}{}'.format('\ns_mass_unit = ', s_mass_unit, ', s_position_unit = ', s_position_unit, ', s_velocity_unit = ', s_velocity_unit)))
    file.write('\n\nNote: If the above units are blank, this means that the soliton parameters were specified in code units')
    file.close()

    loc = save_path + '/' + timestamp



    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID

    gridvec = np.linspace(-gridlength / 2.0 + gridlength / float(2 * resol), gridlength / 2.0 - gridlength / float(2 * resol), resol)

    xarray = np.ones((resol, 1, 1))
    yarray = np.ones((1, resol, 1))
    zarray = np.ones((1, 1, resol))

    xarray[:, 0, 0] = gridvec
    yarray[0, :, 0] = gridvec
    zarray[0, 0, :] = gridvec

    distarray = ne.evaluate("(xarray**2+yarray**2+zarray**2)**0.5") # Radial coordinates



    ##########################################################################################
    # SET UP K-SPACE COORDINATES FOR COMPLEX DFT (NOT RHO DFT)

    kvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    kxarray = np.ones((resol, 1, 1))
    kyarray = np.ones((1, resol, 1))
    kzarray = np.ones((1, 1, resol))
    kxarray[:, 0, 0] = kvec
    kyarray[0, :, 0] = kvec
    kzarray[0, 0, :] = kvec
    karray2 = ne.evaluate("kxarray**2+kyarray**2+kzarray**2")



    ##########################################################################################
    # INITIALISE SOLITONS WITH SPECIFIED MASS, POSITION, VELOCITY, PHASE

    f = np.load('./Soliton Profile Files/initial_f.npy')

    delta_x = 0.00001 # Needs to match resolution of soliton profile array file. Default = 0.00001

    warn = 0

    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')
    funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')

    for k in range(len(solitons)):
        if (k != 0):
            if (not overlap_check(solitons[k], solitons[:k])):
                warn = 1
            else:
                warn = 0

    for s in solitons:
        mass = convert(s[0], s_mass_unit, 'm')
        position = convert(np.array(s[1]), s_position_unit, 'l')
        velocity = convert(np.array(s[2]), s_velocity_unit, 'v')
        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
        alpha = (mass / 3.883) ** 2
        beta = 2.454
        phase = s[3]
        funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x)
        ####### Impart velocity to solitons in Galilean invariant way
        velx = velocity[0]
        vely = velocity[1]
        velz = velocity[2]
        funct = ne.evaluate("exp(1j*(alpha*beta*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
        psi = ne.evaluate("psi + funct")

    rho = ne.evaluate("real(abs(psi)**2)")

    fft_psi = pyfftw.builders.fftn(psi, axes=(0, 1, 2), threads=num_threads)
    ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)



    ##########################################################################################
    # COMPUTE SIZE OF TIMESTEP (CAN BE INCREASED WITH step_factor)

    delta_t = (gridlength/float(resol))**2/np.pi

    min_num_steps = t / delta_t
    min_num_steps_int = int(min_num_steps + 1)
    min_num_steps_int = int(min_num_steps_int/step_factor)

    if save_number >= min_num_steps_int:
        actual_num_steps = save_number
        its_per_save = 1
    else:
        rem = min_num_steps_int % save_number
        actual_num_steps = min_num_steps_int + save_number - rem
        its_per_save = actual_num_steps / save_number

    h = t / float(actual_num_steps)



    ##########################################################################################
    # SETUP K-SPACE FOR RHO (REAL)

    rkvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    krealvec = 2 * np.pi * np.fft.rfftfreq(resol, gridlength / float(resol))
    rkxarray = np.ones((resol, 1, 1))
    rkyarray = np.ones((1, resol, 1))
    rkzarray = np.ones((1, 1, int(resol / 2) + 1))  # last dimension smaller because of reality condition
    rkxarray[:, 0, 0] = rkvec
    rkyarray[0, :, 0] = rkvec
    rkzarray[0, 0, :] = krealvec
    rkarray2 = ne.evaluate("rkxarray**2+rkyarray**2+rkzarray**2")

    rfft_rho = pyfftw.builders.rfftn(rho, axes=(0, 1, 2), threads=num_threads)
    phik = rfft_rho(rho)  # not actually phik but phik is defined in next line
    phik = ne.evaluate("-4*3.141593*phik/rkarray2")
    phik[0, 0, 0] = 0
    irfft_phi = pyfftw.builders.irfftn(phik, axes=(0, 1, 2), threads=num_threads)



    ##########################################################################################
    # COMPUTE INTIAL VALUE OF POTENTIAL

    phisp = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
    phisp = irfft_phi(phik)
    phisp = ne.evaluate("phisp-(cmass)/distarray")



    ##########################################################################################
    # PRE-LOOP ENERGY CALCULATION

    if (save_options[3]):
        egylist = []
        egpcmlist = []
        egpsilist = []
        ekandqlist = []
        mtotlist = []
        egyarr = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')

        egyarr = ne.evaluate('real((abs(psi))**2)')
        egyarr = ne.evaluate('real((-cmass/distarray)*egyarr)')
        egpcmlist.append(Vcell * np.sum(egyarr))
        tot = Vcell * np.sum(egyarr)

        egyarr = ne.evaluate('real(0.5*(phisp+(cmass)/distarray)*real((abs(psi))**2))')
        egpsilist.append(Vcell * np.sum(egyarr))
        tot = tot + Vcell * np.sum(egyarr)

        funct = fft_psi(psi)
        funct = ne.evaluate('-karray2*funct')
        #ifft_calc = pyfftw.builders.ifftn(calc, axes=(0, 1, 2), threads=num_threads)
        funct = ifft_funct(funct)
        egyarr = ne.evaluate('real(-0.5*conj(psi)*funct)')
        ekandqlist.append(Vcell * np.sum(egyarr))
        tot = tot + Vcell * np.sum(egyarr)

        egylist.append(tot)

        egyarr = ne.evaluate('real((abs(psi))**2)')
        mtotlist.append(Vcell * np.sum(egyarr))



    ##########################################################################################
    # PRE-LOOP SAVE I.E. INITIAL CONFIG

    if (save_options[0]):
        if (npy):
            file_name = "rho_#{0}.npy".format(0)
            np.save(os.path.join(os.path.expanduser(loc), file_name), rho)
        if (npz):
            file_name = "rho_#{0}.npz".format(0)
            np.savez(os.path.join(os.path.expanduser(loc), file_name), rho)
        if (hdf5):
            file_name = "rho_#{0}.hdf5".format(0)
            file_name = os.path.join(os.path.expanduser(loc), file_name)
            f = h5py.File(file_name, 'w')
            dset = f.create_dataset("init", data=rho)
            f.close()
    if (save_options[2]):
        plane = rho[:, :, int(resol / 2)]
        if (npy):
            file_name = "plane_#{0}.npy".format(0)
            np.save(os.path.join(os.path.expanduser(loc), file_name), plane)
        if (npz):
            file_name = "plane_#{0}.npz".format(0)
            np.savez(os.path.join(os.path.expanduser(loc), file_name), plane)
        if (hdf5):
            file_name = "plane_#{0}.hdf5".format(0)
            file_name = os.path.join(os.path.expanduser(loc), file_name)
            f = h5py.File(file_name, 'w')
            dset = f.create_dataset("init", data=plane)
            f.close()
    if (save_options[1]):
        if (npy):
            file_name = "psi_#{0}.npy".format(0)
            np.save(os.path.join(os.path.expanduser(loc), file_name), psi)
        if (npz):
            file_name = "psi_#{0}.npz".format(0)
            np.savez(os.path.join(os.path.expanduser(loc), file_name), psi)
        if (hdf5):
            file_name = "psi_#{0}.hdf5".format(0)
            file_name = os.path.join(os.path.expanduser(loc), file_name)
            f = h5py.File(file_name, 'w')
            dset = f.create_dataset("init", data=psi)
            f.close()
    if (save_options[4]):
        line = rho[:, int(resol / 2), int(resol / 2)]
        file_name2 = "line_#{0}.npy".format(0)
        np.save(os.path.join(os.path.expanduser(loc), file_name2), line)
        


    ##########################################################################################
    # LOOP NOW BEGINS

    halfstepornot = 1  # 1 for a half step 0 for a full step

    tenth = float(save_number/10) #This parameter is used if energy outputs are saved while code is running.
    # See commented section below (line 585)

    clear_output()
    print("The total number of steps is %.0f" % actual_num_steps)
    if warn == 1:
        print("WARNING: Significant overlap between solitons in initial conditions")
    print('\n')
    tinit = time.time()

    for ix in range(actual_num_steps):
        if halfstepornot == 1:
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            halfstepornot = 0
        else:
            psi = ne.evaluate("exp(-1j*h*phisp)*psi")
        funct = fft_psi(psi)
        funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")
        psi = ifft_funct(funct)
        rho = ne.evaluate("real(abs(psi)**2)")
        phik = rfft_rho(rho)  # not actually phik but phik is defined on next line
        phik = ne.evaluate("-4*3.141593*(phik)/rkarray2")
        phik[0, 0, 0] = 0
        phisp = irfft_phi(phik)
        phisp = ne.evaluate("phisp-(cmass)/distarray")

        #Next if statement ensures that an extra half step is performed at each save point
        if (((ix + 1) % its_per_save) == 0):
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            rho = ne.evaluate("real(abs(psi)**2)")
            halfstepornot = 1

        #Next block calculates the energies at each save, not at each timestep.
            if (save_options[3]):

                # Gravitational potential energy density associated with the central potential
                egyarr = ne.evaluate('real((abs(psi))**2)')
                egyarr = ne.evaluate('real((-cmass/distarray)*egyarr)')
                egpcmlist.append(Vcell * np.sum(egyarr))
                tot = Vcell * np.sum(egyarr)

                # Gravitational potential energy density of self-interaction of the condensate
                egyarr = ne.evaluate('real(0.5*(phisp+(cmass)/distarray)*real((abs(psi))**2))')
                egpsilist.append(Vcell * np.sum(egyarr))
                tot = tot + Vcell * np.sum(egyarr)

                funct = fft_psi(psi)
                funct = ne.evaluate('-karray2*funct')
                funct = ifft_funct(funct)
                egyarr = ne.evaluate('real(-0.5*conj(psi)*funct)')
                ekandqlist.append(Vcell * np.sum(egyarr))
                tot = tot + Vcell * np.sum(egyarr)

                egylist.append(tot)

                egyarr = ne.evaluate('real((abs(psi))**2)')
                mtotlist.append(Vcell * np.sum(egyarr))


        #Uncomment next section if partially complete energy lists desired as simulation runs.
        #In this way, some energy data will be saved even if the simulation is terminated early.

                # if (ix+1) % tenth == 0:
                #     label = (ix+1)/tenth
                #     file_name = "{}{}".format(label,'egy_cumulative.npy')
                #     np.save(os.path.join(os.path.expanduser(loc), file_name), egylist)
                #     file_name = "{}{}".format(label,'egpcm_cumulative.npy')
                #     np.save(os.path.join(os.path.expanduser(loc), file_name), egpcmlist)
                #     file_name = "{}{}".format(label,'egpsi_cumulative.npy')
                #     np.save(os.path.join(os.path.expanduser(loc), file_name), egpsilist)
                #     file_name = "{}{}".format(label,'ekandq_cumulative.npy')
                #     np.save(os.path.join(os.path.expanduser(loc), file_name), ekandqlist)


        ################################################################################
        # SAVE DESIRED OUTPUTS

        if (save_options[0] and ((ix + 1) % its_per_save) == 0):
            if (npy):
                file_name = "rho_#{0}.npy".format(int((ix + 1) / its_per_save))
                np.save(os.path.join(os.path.expanduser(loc), file_name), rho)
            if (npz):
                file_name = "rho_#{0}.npz".format(int((ix + 1) / its_per_save))
                np.savez(os.path.join(os.path.expanduser(loc), file_name), rho)
            if (hdf5):
                file_name = "rho_#{0}.hdf5".format(int((ix + 1) / its_per_save))
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=rho)
                f.close()
        if (save_options[2] and ((ix + 1) % its_per_save) == 0):
            plane = rho[:, :, int(resol / 2)]
            if (npy):
                file_name = "plane_#{0}.npy".format(int((ix + 1) / its_per_save))
                np.save(os.path.join(os.path.expanduser(loc), file_name), plane)
            if (npz):
                file_name = "plane_#{0}.npz".format(int((ix + 1) / its_per_save))
                np.savez(os.path.join(os.path.expanduser(loc), file_name), plane)
            if (hdf5):
                file_name = "plane_#{0}.hdf5".format(int((ix + 1) / its_per_save))
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=plane)
                f.close()
        if (save_options[1] and ((ix + 1) % its_per_save) == 0):
            if (npy):
                file_name = "psi_#{0}.npy".format(int((ix + 1) / its_per_save))
                np.save(os.path.join(os.path.expanduser(loc), file_name), psi)
            if (npz):
                file_name = "psi_#{0}.npz".format(int((ix + 1) / its_per_save))
                np.savez(os.path.join(os.path.expanduser(loc), file_name), psi)
            if (hdf5):
                file_name = "psi_#{0}.hdf5".format(int((ix + 1) / its_per_save))
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=psi)
                f.close()
        if (save_options[4] and ((ix + 1) % its_per_save) == 0):
            line = rho[:, int(resol/2), int(resol / 2)]
            file_name2 = "line_#{0}.npy".format(int((ix + 1) / its_per_save))
            np.save(os.path.join(os.path.expanduser(loc), file_name2), line)



        ################################################################################
        # UPDATE INFORMATION FOR PROGRESS BAR

        tint = time.time() - tinit
        tinit = time.time()
        prog_bar(actual_num_steps, ix + 1, tint)



    ################################################################################
    # LOOP ENDS

    clear_output()
    print ('\n')
    print("Complete.")
    if warn == 1:
        print("WARNING: Significant overlap between solitons in initial conditions")

    if (save_options[3]):
        file_name = "egylist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egylist)
        file_name = "egpcmlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpcmlist)
        file_name = "egpsilist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpsilist)
        file_name = "ekandqlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), ekandqlist)
        file_name = "masslist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), mtotlist)

