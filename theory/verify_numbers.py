"""Reproduce the 2026-09-24 scientific audit, without running the legacy notebook.
Run from any directory: python3 scripts/audit_revised_manuscript.py
Writes results/theory_checks.json.
"""
from pathlib import Path
import json
import re
import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
checks = []
def close(name, actual, expected, **kw):
    assert np.allclose(actual, expected, **kw), (name, actual, expected)
    checks.append(name)

def eshelby(alpha, nu):
    axes2 = [1., 1., alpha**2]
    def integral(i, j=None):
        integrand = lambda q: 1/((axes2[i]+q)*np.sqrt(np.prod(np.array(axes2)+q))
                         *(1 if j is None else axes2[j]+q))
        bounds = sorted(set([0., alpha**2, 1.])) + [np.inf]
        return 2*np.pi*alpha*sum(quad(integrand, lo, hi, epsabs=1e-10)[0]
                                   for lo, hi in zip(bounds[:-1], bounds[1:]))
    ii = np.array([integral(i) for i in range(3)])
    ij = np.array([[integral(i,j) for j in range(3)] for i in range(3)])
    s13 = ((1+alpha**2)*ij[0,2]+(1-2*nu)*(ii[0]+ii[2]))/(16*np.pi*(1-nu))
    # Normal-normal Eshelby components, including non-major-symmetric terms.
    sn = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            sn[i,j] = ((3*axes2[i]*ij[i,i]+(1-2*nu)*ii[i]) if i==j else
                       (axes2[j]*ij[i,j]-(1-2*nu)*ii[i]))/(8*np.pi*(1-nu))
    return ii,ij,s13,sn

nu=1/3
iv, ijv, s13, sn = eshelby(.5,nu)
e=np.sqrt(.75)
i1_closed=2*np.pi*.5/e**3*(np.arccos(.5)-.5*e)
close('I1 quadrature versus closed form',iv[0],i1_closed)
close('I sum rule',iv.sum(),4*np.pi)
close('I13 identity',ijv[0,2],(iv[2]-iv[0])/.75)
close('sphere shear limit',eshelby(1,nu)[2],(4-5*nu)/(15*(1-nu)))
alpha_thin=.001
e_thin=np.sqrt(1-alpha_thin**2)
i1_thin=2*np.pi*alpha_thin/e_thin**3*(np.arccos(alpha_thin)-alpha_thin*e_thin)
i3_thin=4*np.pi-2*i1_thin
i13_thin=(i3_thin-i1_thin)/(1-alpha_thin**2)
s13_thin=((1+alpha_thin**2)*i13_thin+(1-2*nu)*(i1_thin+i3_thin))/(16*np.pi*(1-nu))
close('thin disk shear limit (closed form)',s13_thin,.5,atol=.002)
for alpha in [.2,.5,1.]:
    close(f'dilatational trace shape independence {alpha}',eshelby(alpha,nu)[3].sum(),(1+nu)/(1-nu))
beta1=1-2*s13;beta2=4*(1+nu)/(9*(1-nu))
# Symbolically differentiate the actual Kelvin displacement, independently of prose.
x,y,z,G,g,V,n=sp.symbols('x y z G g V n', real=True)
r=sp.sqrt(x*x+y*y+z*z);A=1/(16*sp.pi*G*(1-n));M=G*g*V
u1=-M*A*(2*(2*n-1)*z/r**3-6*x*x*z/r**5)
u3=-M*A*(2*(2*n-1)*x/r**3-6*x*z*z/r**5)
stress=G*(sp.diff(u1,z)+sp.diff(u3,x))
T=sp.Rational(1,2)-n+3*n*(x*x+z*z)/(2*r*r)-sp.Rational(15,2)*x*x*z*z/r**4
assert sp.simplify(stress-G*g*V*T/(2*sp.pi*(1-n)*r**3))==0
checks.append('Kelvin differentiation and angular shear factor')
close('negative diagonal shear lobe',.5-nu+1.5*nu-7.5/4,-29/24)
# Independent numerical Cauchy principal value for a unit Lorentzian.
x0=.7; ww=1.2
q=quad(lambda t: ww/(np.pi*(t*t+ww*ww)), -10000,10000,
       weight='cauchy',wvar=x0,epsabs=1e-10,limit=500)[0]
close('Lorentzian Hilbert transform',-q,x0/(x0*x0+ww*ww),atol=1e-8)
close('Lorentzian normalisation',quad(lambda t:ww/(np.pi*(t*t+ww*ww)),-np.inf,np.inf)[0],1)
close('normalised disk kernel',quad(lambda t:ww*t/(t*t+ww*ww)**1.5,0,np.inf)[0],1)
# Projection of regularised stress numerator.
a2=x0*x0+ww*ww
q=quad(lambda yy:((1+nu)*x0*x0+(1-2*nu)*yy*yy)/(a2+yy*yy)**2.5,-np.inf,np.inf)[0]
close('regularised traction projection',q,4*(1+nu)*x0*x0/(3*a2*a2)+2*(1-2*nu)/(3*a2))
eV=1.602176634e-19;k=1.380649e-23;W=.75e-9;V0=2*np.pi/3*W**3
energy=.5*(beta1*.1**2+beta2*.05**2)*22e9*V0/eV
bias=5e6*.1*V0/2/eV; temp=773.; ratio=bias*eV/(k*temp)
ratefactor=2*W*.1/1e-5*np.exp(-energy*eV/(k*temp))*np.sinh(ratio)
fr=48.3e9*.2556e-9/(2*np.pi*(1-nu)*1e-6)
rows=json.loads((ROOT/'theory'/'data'/'systems_41.json').read_text())
a=np.array([r['values'] for r in rows]);assert len(a)==41
qr=a[:,3]/a[:,2];br=a[:,4]/a[:,5]
summary={'checks_passed':checks,'I1':iv[0],'I3':iv[2],'W2_I13':ijv[0,2],
 'S1313':s13,'beta1':beta1,'beta2':beta2,'volume_nm3':V0*1e27,
 'elastic_energy_eV':energy,'reference_energy_relative_error_percent':100*(energy/.38-1),
 'volumetric_fraction':beta2*.05**2/(beta1*.1**2+beta2*.05**2),
 'bias_eV':bias,'bias_over_kT':ratio,'rate_per_eta_attempt_frequency':ratefactor,
 'eta_attempt_frequency_for_0.16_per_s':.16/ratefactor,'FR_MPa':fr/1e6,
 'strain_means':a[:,:2].mean(0).tolist(),'strain_sample_sd':a[:,:2].std(0,ddof=1).tolist(),
 'shear_0.1_relative_error_percent':100*(.1/a[:,0].mean()-1),
 'shear_0.12_relative_error_percent':100*(.12/a[:,0].mean()-1),
 'dilatation_relative_error_percent':100*(.05/a[:,1].mean()-1),
 'Q_over_elastic_mean':qr.mean(),'Q_over_elastic_sample_sd':qr.std(ddof=1),
 'GB_ratio_median':np.median(br),'GB_ratio_min':br.min(),'GB_ratio_max':br.max(),
 'GB_ratio_within_3_5_10':[int(((br<=v)&(br>=1/v)).sum()) for v in [3,5,10]],
 'provenance':'Statistics use rounded rows transcribed in the previous SI, not an independent experimental dataset.'}
# Tensor-strain convention of the boundary-sliding model (Harisankar and Padmanabhan 2025)
summary['beta1_tensor']=4*beta1
summary['beta1_HP2025']=0.944*(1.590-nu)/(1-nu)
fac=(4*beta1*a[:,0]**2+beta2*a[:,1]**2)/(beta1*a[:,0]**2+beta2*a[:,1]**2)
qt=a[:,3]/(a[:,2]*fac)
summary['tensor_energy_factor_mean']=float(fac.mean())
summary['Q_over_elastic_tensor_mean']=float(qt.mean()); summary['Q_over_elastic_tensor_sd']=float(qt.std(ddof=1))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'theory_checks.json').write_text(json.dumps(summary,indent=2,default=float)+'\n')
print(json.dumps(summary,indent=2,default=float))
