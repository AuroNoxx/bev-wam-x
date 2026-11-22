"""Ordinary navigation from a directed road sketch, GNSS and an IMU compass.
This module is shared by live driving, collection and simulator-free replay.
"""
import numpy as np


def route_vertices(spec):
    vertices = np.asarray(spec['xy'], dtype=float)
    laps = spec.get('laps') or 1
    if laps == 1:
        return vertices
    if not isinstance(laps, int) or laps < 1 or not np.array_equal(vertices[0], vertices[-1]):
        raise ValueError('Repeated laps require a closed SD route and a positive integer')
    return np.vstack([vertices[:-1]] * laps + [vertices[-1:]])


class RoadNavigator:
    def __init__(self,vertices,georef,phase=0.,events=(),accuracy=None):
        self.vertices=np.asarray(vertices,float)
        self.delta=np.diff(self.vertices,axis=0)
        self.length=np.linalg.norm(self.delta,axis=1)
        if np.any(self.length<.1) or not len(self.length):raise ValueError('Invalid SD route')
        self.arc=np.r_[0.,self.length.cumsum()]
        self.direction=self.delta/self.length[:,None]
        fixes=np.asarray(georef,float)
        self.origin=fixes[0]
        self.to_xy=3000*np.linalg.inv(fixes[1:3]-self.origin)
        if np.linalg.norm((fixes[3]-self.origin)@self.to_xy-[3000,3000])>2:
            raise ValueError('Survey GNSS registration has excessive residual')
        self.station=None
        self.phase=phase
        self.events=events
        self.accuracy=np.asarray(accuracy,float) if accuracy is not None else None

    def update(self,lat,lon,compass,seconds):
        if not np.isfinite([lat,lon,compass,seconds]).all():raise ValueError('Nonfinite localization')
        forward=np.array([np.sin(compass),-np.cos(compass)])
        left=np.array([forward[1],-forward[0]])
        measured=(np.array([lat,lon])-self.origin)@self.to_xy
        self.gnss_xy=measured
        amplitude=2.5
        if self.accuracy is not None:
            # Coarse transverse margin surveyed from the road surface. Include the
            # full preview and smooth the bias, including on approaches to narrows.
            station=0 if self.station is None else self.station
            sample=np.arange(station-5,station+76,2)
            cap=np.interp(sample,self.accuracy[:,0],self.accuracy[:,1])
            amplitude=float(np.min(cap))
        self.drift=amplitude*np.sin(self.phase+.12*seconds)
        self.xy=measured+left*self.drift
        self.axes=np.stack((forward,left),axis=1)
        eligible=self.direction@forward>.5
        if self.station is None:
            eligible &= self.arc[:-1]<160
        else:
            eligible &= (self.arc[:-1]<self.station+18)&(self.arc[1:]>self.station-10)
        indices=np.flatnonzero(eligible)
        if not len(indices):raise ValueError('No forward SD road segment in the local progress window')
        offset=self.xy-self.vertices[indices]
        along=np.clip((offset*self.direction[indices]).sum(1),0,self.length[indices])
        dist=np.linalg.norm(offset-self.direction[indices]*along[:,None],axis=1)
        cost=dist+(0 if self.station is None else .035*abs(self.arc[indices]+along-self.station))
        best=cost.argmin()
        self.station=float(self.arc[indices[best]]+along[best])
        self.error=float(dist[best])
        return self.points([20,40,60])

    def points(self,offsets):
        d=np.minimum(self.arc[-1],self.station+np.asarray(offsets))
        xy=np.stack([np.interp(d,self.arc,v) for v in self.vertices.T],axis=1)
        return ((xy-self.xy)@self.axes).astype(np.float32)

    def instruction(self):
        upcoming=[(s-self.station,k) for s,k in self.events if s>=self.station-3]
        return min(upcoming) if upcoming else (-1.,'follow')


def fix_values(gnss,imu):
    return [gnss.latitude,gnss.longitude,gnss.altitude,imu.compass,
            imu.accelerometer.x,imu.accelerometer.y,imu.accelerometer.z,
            imu.gyroscope.x,imu.gyroscope.y,imu.gyroscope.z]

