from pathlib import Path

import hashlib

import json

import math

from datetime import datetime, timezone

import numpy as np

from scipy.io import savemat

def se(q, ell):
    d = (q[:, None, :] - q[None, :, :]) / np.asarray(ell)
    return np.exp(-.5 * np.sum(d*d, axis=2))

def matern(q, ell, order):
    r = np.abs(q[:, None, :] - q[None, :, :]) / np.asarray(ell)
    if order == 1:
        factors = np.exp(-r)
    elif order == 3:
        a = np.sqrt(3.) * r
        factors = (1 + a) * np.exp(-a)
    elif order == 5:
        a = np.sqrt(5.) * r
        factors = (1 + a + a*a/3) * np.exp(-a)
    else:
        raise ValueError(order)
    return factors.prod(axis=2)

def periodic(q, ell, period):
    d = q[:, None, :] - q[None, :, :]
    return np.exp(-2 * np.sum(np.sin(np.pi*d/period)**2 / np.asarray(ell)**2, axis=2))

def geometry(x):
    center = np.array([np.mean(x[:, 0]), 0.])
    scale = x.std(axis=0, ddof=1)
    return (x-center)/scale, center, scale

def make_case(index):
    seed = 202610080 + index
    rng = np.random.default_rng(seed)
    if index % 2:
        uv = rng.uniform(size=(96, 2))
        layout = 'scattered'
    else:
        horizontal = np.array([.04, .19, .38, .57, .78, .96])
        depths = (np.arange(16)+.5)/16
        uv = np.column_stack([np.repeat(horizontal,16),np.tile(depths,6)])
        uv[:,1] += rng.uniform(-.012,.012,96)
        layout = 'six_boreholes'
    x = uv * np.array([180.,18.])
    if index == 4:
        x[:,0] += 500.
    q, center, scale = geometry(x)
    u,v = uv.T
    if index == 1:
        mean = 4 + .8*u - .4*v
        K = .9**2 * se(q,[.65,1.1]); noise = .14
        truth = dict(mechanism='anisotropic_SE', ell=[.65,1.1], signal_sd=.9, mean='4 + .8*u - .4*v', in_library=True)
    elif index == 2:
        mean = np.full(96,3.5)
        K = .8**2 * matern(q,[.55,.9],1); noise = .16
        truth = dict(mechanism='product_Matern12', ell=[.55,.9], signal_sd=.8, mean='constant 3.5', in_library=True)
    elif index == 3:
        mean = 4.2 + .45*u - .25*v + .5*u*u - .3*v*v
        K = .65**2 * se(q,[1.1,.7]) + .5**2 * periodic(q,[1.2,1.2],1.65); noise=.12
        truth = dict(mechanism='SE_plus_PER', se_ell=[1.1,.7], se_sd=.65, per_ell=[1.2,1.2], per_sd=.5, period=1.65, mean='4.2+.45*u-.25*v+.5*u^2-.3*v^2', in_library=True)
    elif index == 4:
        mean = 3.8 + .5*u + .35*v
        shifted = q.copy(); shifted[:,0] -= -.45
        phi = shifted/np.array([1.8,3.2])
        K = matern(q,[.85,.65],3) * (phi@phi.T); noise=.13
        truth = dict(mechanism='Matern32_times_shifted_LIN', matern_ell=[.85,.65], lin_scales=[1.8,3.2], lin_shift=-.45, mean='3.8+.5*u+.35*v', in_library=True)
    elif index == 5:
        mean = 4.5 + .3*u - .2*v
        warp = q.copy(); warp[:,0] = q[:,0] + .18*q[:,0]**2
        amplitude = .55 + .55*u
        K = matern(q,[1.25,.95],5) * periodic(warp,[1.1,1.5],1.55)
        K *= amplitude[:,None]*amplitude[None,:]; noise=.14
        truth = dict(mechanism='warped_quasiperiodic_amplitude', matern_ell=[1.25,.95], per_ell=[1.1,1.5], period=1.55, warp='q1+.18*q1^2', amplitude='.55+.55*u', mean='4.5+.3*u-.2*v', in_library=False)
    else:
        mean = 4 + .4*u + .6*np.tanh(7*(v-.55))
        gate = 1/(1+np.exp(-10*(u-.5)))
        angle = .45
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        qr = q@rotation.T
        K1 = .7**2*se(qr,[1.3,.55]); K2 = .95**2*matern(q,[.4,.85],3)
        K = (1-gate[:,None])*(1-gate[None,:])*K1 + gate[:,None]*gate[None,:]*K2
        noise=.16
        truth = dict(mechanism='smooth_two_regime_covariance', gate='sigmoid(10*(u-.5))', first='rotated SE, sd .7, ell [1.3,.55], rotation .45', second='Matern32, sd .95, ell [.4,.85]', mean='4+.4*u+.6*tanh(7*(v-.55))', in_library=False)
    K = (K+K.T)/2
    eigen_min = float(np.linalg.eigvalsh(K)[0])
    if eigen_min < -1e-9 or not np.isfinite(K).all():
        raise ValueError('INVALID_GENERATOR_COVARIANCE')
    latent_jitter = 1e-12*max(1.,float(np.max(np.diag(K))))
    latent = mean + np.linalg.cholesky(K+latent_jitter*np.eye(96))@rng.standard_normal(96)
    y = latent + noise*rng.standard_normal(96)
    truth.update(seed=seed, layout=layout, noise_sd=noise, geometry_center=center.tolist(), geometry_scale=scale.tolist(), covariance_min_eigenvalue=eigen_min, latent_jitter=latent_jitter)
    return x,y,truth
