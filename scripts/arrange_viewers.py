"""Evaluation launcher layout: project dashboard left, project CARLA right."""
from pathlib import Path
import ctypes
from ctypes import wintypes as W
import json
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
user=ctypes.WinDLL('user32',use_last_error=True)
user.SetThreadDpiAwarenessContext.argtypes=[W.HANDLE]
user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
user.GetWindowThreadProcessId.argtypes=[W.HWND,ctypes.POINTER(W.DWORD)]
user.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,ctypes.c_int]
user.IsWindowVisible.argtypes=[W.HWND]
user.MonitorFromWindow.argtypes=[W.HWND,W.DWORD];user.MonitorFromWindow.restype=W.HANDLE
user.SetWindowPos.argtypes=[W.HWND,W.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,W.UINT]
user.ShowWindow.argtypes=[W.HWND,ctypes.c_int]
class MonitorInfo(ctypes.Structure):
    _fields_=[('size',W.DWORD),('monitor',W.RECT),('work',W.RECT),('flags',W.DWORD)]
user.GetMonitorInfoW.argtypes=[W.HANDLE,ctypes.POINTER(MonitorInfo)]
Callback=ctypes.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
user.EnumWindows.argtypes=[Callback,W.LPARAM]

def find():
    processes={p.pid:p.info for p in psutil.process_iter(['pid','name','exe'])}
    windows={}
    @Callback
    def visit(hwnd,_):
        if not user.IsWindowVisible(hwnd):return True
        pid=W.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        info=processes.get(pid.value,{})
        title=ctypes.create_unicode_buffer(512);user.GetWindowTextW(hwnd,title,512)
        if info.get('name')=='CarlaUnreal-Win64-Shipping.exe' and info.get('exe') and Path(info['exe']).is_relative_to(ROOT/'runtime'):
            windows['carla']=hwnd
        if info.get('name') in ('chrome.exe','msedge.exe','firefox.exe') and title.value.startswith('Highway OCC ·'):
            windows['web']=hwnd
        return True
    user.EnumWindows(visit,0)
    return windows

def main():
    for _ in range(60):
        windows=find()
        if len(windows)==2:break
        time.sleep(.5)
    else:
        print('No matching dashboard/CARLA pair found. Snap manually using Win+Left / Win+Right.');return
    info=MonitorInfo();info.size=ctypes.sizeof(info)
    if not user.GetMonitorInfoW(user.MonitorFromWindow(windows['carla'],2),ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    r=info.work;half=(r.right-r.left)//2;height=r.bottom-r.top
    for name,x,width in [('web',r.left,half),('carla',r.left+half,r.right-r.left-half)]:
        user.ShowWindow(windows[name],9)
        if not user.SetWindowPos(windows[name],None,x,r.top,width,height,0x0010|0x0004):
            raise ctypes.WinError(ctypes.get_last_error())
    (ROOT/'logs/occ-live/layout.json').write_text(json.dumps({'windows':windows,'work_area':[r.left,r.top,r.right,r.bottom],'web':'left','carla':'right'}),encoding='utf-8')

if __name__=='__main__':main()
