import numpy as np
import matplotlib.pyplot as plt

class StarModel:
    def initial_n(self, rho_s, mn):
        
        '''
        The initial Density of a Neutron Star
        '''
        
        n = 1
        err = 1
        tol = 1e-15
        count = 0
        
        while err > tol:
            count += 1
            fn = 236*n**(2.54) + n * mn - rho_s
            dfn = 236*2.54*n**(1.54) +mn
            temp = n - fn / dfn
            err = np.abs(n-temp)
            n = temp
        #print(f"Newton-Raphson Converged after {count} iterations")
        return n

    def rho(self, p, rho_s, mn):
        
        '''
        Energy Density of a neutron star at a given pressure
        rho_s - Central density at r = 0
        mn = mass of a neutron
        n - number density at given pressure
        '''
        
        n = (p*rho_s/363.44)**(1/2.54)
        return (236. * n**2.54 + n * mn)/rho_s

    def dp_dr(self, r, m, p, rho_s, mn, flag):
        
        '''
        Pressure Gradient
        r - the distance from the center of the star (radius)
        m - Mass enclosed withing the value of r
        p - pressure at that value of r
        flag - Classical or Relativisitic model
        '''
        
        if flag == 0:
            #Classical Model
            y = -m*self.rho(p, rho_s, mn)/(r**2 + 1e-20)
        else:
            #Relativistic Model
            y = -(p+self.rho(p, rho_s, mn)) * (m + p*r**3)/(r**2 - 2*m*r + 1e-20)
            
        return y

    def dm_dr(self, r, m, p, rho_s, mn):
        
        '''
        Mass gradient
        r - the distance from the center of the star (radius)
        m - Mass enclosed withing the value of r
        p - pressure at that value of r
        flag - Classical or Relativisitic model
        '''
        
        return self.rho(p, rho_s, mn)*r**2

    def RK4Solver(self, r, m, p, h, rho_s, mn, flag):
        
        '''
        Runge - Kutta 4th order Method
        Calculates 4 different pressure and mass gradients
        K1_ - Mass gradients
        K2_ - Pressure Gradients
        '''
        
        y = np.zeros(2)
        
        #Gradient 1 at start point
        k11 = self.dm_dr(r, m, p, rho_s, mn)
        k21 = self.dp_dr(r, m, p, rho_s, mn, flag)
        
        #Gradient 2 at mid of start and end point
        k12 = self.dm_dr(r + 0.5*h, m + 0.5*k11*h, p + 0.5*k21*h, rho_s, mn)
        k22 = self.dp_dr(r + 0.5*h, m + 0.5*k11*h, p + 0.5*k21*h, rho_s, mn, flag)
        
        #Gradient 3 at mid of start and end point
        k13 = self.dm_dr(r + 0.5*h, m + 0.5*k12*h, p + 0.5*k22*h, rho_s, mn)
        k23 = self.dp_dr(r + 0.5*h, m + 0.5*k12*h, p + 0.5*k22*h, rho_s, mn, flag)
        
        #Gradient 4 at start point
        k14 = self.dm_dr(r + h, m + k13*h, p + k23*h, rho_s, mn)
        k24 = self.dp_dr(r + h, m + k13*h, p + k23*h, rho_s, mn, flag)
        
        #Update of m and p
        y[0] = m + h*(k11 + 2*k12 + 2*k13 + k14)/6
        y[1] = p + h*(k21 + 2*k22 + 2*k23 + k24)/6
        
        return y

    def plot_data(self, color, label, r, R0, m, M0, Ms):
        
        #Mass profile
        plt.subplot(1, 2, 1)
        plt.plot(r*R0*1e-18, m*M0/Ms, color = color, linewidth = 1.2, label = label)
        plt.xlabel('Distance, $r$ (km)', fontsize = 13)
        plt.ylabel('Mass, $M/M_{sun}$', fontsize = 13)
        plt.title('Mass Profile of a Neutron Star', color = "tab:red", weight = "bold", fontsize = 15)
        plt.xlim(left = 0)
        plt.ylim(bottom = 0)
        plt.legend(fontsize = 13, frameon = False)
        
        #Mass profile
        plt.subplot(1, 2, 1)
        plt.plot(r*R0*1e-18, m*M0/Ms, color = color, linewidth = 1.2, label = label)
        plt.xlabel('Distance, $r$ (km)', fontsize = 13)
        plt.ylabel('Pressure, $P$ $(MeV/fm^{3})$', fontsize = 13)
        plt.title('Pressure Profile of a Neutron Star', color = "tab:red", weight = "bold", fontsize = 15)
        plt.xlim(left = 0)
        plt.ylim(bottom = 0)
        plt.legend(fontsize = 13, frameon = False)
        
    def run_calculation(self, mass):
        
        #Simulation parameters and Conversion factors
        hc = 197.327                                # conversion factor in MeV fm (h bar * c)
        G = hc * 6.67259e-45                        # gravitational constant in MeV^-1 fm^3 kg^-1
        Ms = 1.1157467e60                           # the mass of the sun in MeV
        rho_s = 1665.3                              # central density of a neutron star (density at r = 0) in MeV/fm^3
        M0 = (4*3.14159265*(G**3)*rho_s)**(-0.5)
        R0 = G*M0
        mn = mass                              # the mass of a neutron in MeV c^-2

        #Initialising values and array
        '''Theese are the values for the radius, the step size and the tolerance value'''
        N = 1501                            # Total number of data points
        r = np.linspace(0,15,N)             # values of radius to compute enclosed mass within it
        h = r[1]-r[0]                       # step size for RK4Solver
        tol = 9e-5                          # tolerance for RK4Solver

        '''Arrays to store the updation values in RK4Solver'''
        m = np.zeros(N)                     # mass
        p = np.zeros(N)                     # pressure

        '''Initial Number Density of Neutrons at r = 0'''
        ni = self.initial_n(rho_s, mn)

        '''Setting initial values in the array - values at the center of the neutron star'''
        r[0] = 0
        m[0] = 0
        p[0] = 363.44 * (ni**2.54)/rho_s

        '''Setting flags for choosing between classical and relativistic model'''
        flag_set = [0,1]

        '''Printing initial values'''
        """print("Initial number density, ni =", ni)
        print("Initial pressure, P[0] =", p[0]*rho_s, "MeV/fm^3")
        print("Simulation range, R = 0 to R =", r[-1]*R0*1e-18, "km")
        print("Step size for RK4 Solver:", h)"""


        '''Using the RK4 Numerical Method for modeling a neutron star'''
        plt.figure(figsize= (18, 5))
        flag = flag_set[1]
        for i in range(0, N-1):
            if flag == 0:
                [m[i+1], p[i+1]] = self.RK4Solver(r[i], m[i], p[i], h, rho_s, mn, flag)
            else:
                [m[i+1], p[i+1]] = self.RK4Solver(r[i], m[i], p[i], h, rho_s, mn, flag)
            if p[i+1] < tol:
                break
        #print()
        if i == N-2:
            lbl1 = "Program didn't converge to P = 0, extend the maximum value of r"
        else:
            lbl1 = f"P < {tol} found after {i} runs"
            
        '''Keep only the used indices of array and discard the remaining ones'''
        m = m[:i+2]
        p = p[:i+2]
        r = r[:i+2]
        
        '''Visualise and print the results'''
        if flag == 0:
            lbl = "Classical Model"
            self.plot_data('tab:orange', "Classical Model", r, R0, m, M0, Ms)
        else:
            lbl = "Relativistic Model"
            self.plot_data('tab:cyan', "Relativistic Model", r, R0, m, M0, Ms)
            
        '''Printing the overall output'''
        """print("=============================================================================")
        print(lbl, "Results:", lbl1)
        print("=============================================================================")
        print("Initial density, rho_s =", rho_s, "MeV/fm^3")
        print("Total mass =", m[-1]*M0/Ms, "times Solar mass")
        print("Radius of the Neutron Star =", r[-1]*R0*1e-18, "km")"""
            
        #print()
        plt.subplots_adjust(wspace=0.15)
        
        return rho_s, m[-1]*M0/Ms, r[-1]*R0*1e-18