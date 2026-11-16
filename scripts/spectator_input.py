"""Non-intercepting Windows mouse deltas scoped to this project's CARLA foreground."""
import ctypes as C
from ctypes import wintypes as W
import threading

U=C.WinDLL('user32',use_last_error=True)
K=C.WinDLL('kernel32',use_last_error=True)
Result=C.c_ssize_t
Proc=C.WINFUNCTYPE(Result,W.HWND,W.UINT,W.WPARAM,W.LPARAM)
class WindowClass(C.Structure):
    _fields_=[('style',W.UINT),('proc',Proc),('class_extra',C.c_int),('window_extra',C.c_int),('instance',W.HINSTANCE),('icon',W.HICON),('cursor',W.HANDLE),('background',W.HBRUSH),('menu',W.LPCWSTR),('name',W.LPCWSTR)]
class Device(C.Structure):
    _fields_=[('page',W.USHORT),('usage',W.USHORT),('flags',W.DWORD),('target',W.HWND)]
class Header(C.Structure):
    _fields_=[('kind',W.DWORD),('size',W.DWORD),('device',W.HANDLE),('param',W.WPARAM)]
class Mouse(C.Structure):
    _fields_=[('flags',W.USHORT),('buttons',W.ULONG),('raw_buttons',W.ULONG),('dx',W.LONG),('dy',W.LONG),('extra',W.ULONG)]
class Raw(C.Structure):
    _fields_=[('header',Header),('mouse',Mouse)]
K.GetModuleHandleW.argtypes=[W.LPCWSTR];K.GetModuleHandleW.restype=W.HMODULE
U.RegisterClassW.argtypes=[C.POINTER(WindowClass)];U.RegisterClassW.restype=W.ATOM
U.CreateWindowExW.argtypes=[W.DWORD,W.LPCWSTR,W.LPCWSTR,W.DWORD,C.c_int,C.c_int,C.c_int,C.c_int,W.HWND,W.HMENU,W.HINSTANCE,C.c_void_p];U.CreateWindowExW.restype=W.HWND
U.DefWindowProcW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM];U.DefWindowProcW.restype=Result
U.RegisterRawInputDevices.argtypes=[C.POINTER(Device),W.UINT,W.UINT]
U.GetRawInputData.argtypes=[W.HANDLE,W.UINT,C.c_void_p,C.POINTER(W.UINT),W.UINT];U.GetRawInputData.restype=W.UINT
U.GetForegroundWindow.restype=W.HWND
U.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
U.GetMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT]
U.TranslateMessage.argtypes=[C.POINTER(W.MSG)]
U.DispatchMessageW.argtypes=[C.POINTER(W.MSG)];U.DispatchMessageW.restype=Result
U.PostMessageW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM]
U.DestroyWindow.argtypes=[W.HWND]

class MouseLook:
    def __init__(self,pids):
        self.pids=set(pids);self.lock=threading.Lock();self.dx=0;self.dy=0
        self.hwnd=None;self.error=None;self.ready=threading.Event()
        threading.Thread(target=self._run,daemon=True).start()
        self.ready.wait(5)
        if self.error:raise self.error
        if not self.hwnd:raise RuntimeError('Mouse-look input did not initialize')

    def focused(self):
        pid=W.DWORD();U.GetWindowThreadProcessId(U.GetForegroundWindow(),C.byref(pid))
        return pid.value in self.pids

    def sample(self):
        held=self.focused() and bool(U.GetAsyncKeyState(1)&0x8000)
        with self.lock:
            dx,dy=self.dx,self.dy;self.dx=self.dy=0
        return held,dx,dy

    def close(self):
        if self.hwnd:U.PostMessageW(self.hwnd,0x10,0,0)

    def _run(self):
        try:
            @Proc
            def handle(hwnd,message,wparam,lparam):
                if message==0xff and self.focused() and U.GetAsyncKeyState(1)&0x8000:
                    raw=Raw();size=W.UINT(C.sizeof(raw))
                    received=U.GetRawInputData(W.HANDLE(lparam),0x10000003,C.byref(raw),C.byref(size),C.sizeof(Header))
                    if received!=0xffffffff and raw.header.kind==0 and not raw.mouse.flags&1:
                        with self.lock:
                            self.dx+=raw.mouse.dx;self.dy+=raw.mouse.dy
                elif message==0x10:
                    device=Device(1,2,1,None);U.RegisterRawInputDevices(C.byref(device),1,C.sizeof(device))
                    U.DestroyWindow(hwnd);U.PostQuitMessage(0);return 0
                return U.DefWindowProcW(hwnd,message,wparam,lparam)
            self.callback=handle
            instance=K.GetModuleHandleW(None)
            cls=WindowClass(0,handle,0,0,instance,None,None,None,None,'HighwayOccMouseInput')
            if not U.RegisterClassW(C.byref(cls)):raise C.WinError(C.get_last_error())
            self.hwnd=U.CreateWindowExW(0,cls.name,'',0,0,0,0,0,W.HWND(-3),None,instance,None)
            if not self.hwnd:raise C.WinError(C.get_last_error())
            device=Device(1,2,0x100,self.hwnd)
            if not U.RegisterRawInputDevices(C.byref(device),1,C.sizeof(device)):raise C.WinError(C.get_last_error())
            self.ready.set()
            msg=W.MSG()
            while U.GetMessageW(C.byref(msg),None,0,0)>0:
                U.TranslateMessage(C.byref(msg));U.DispatchMessageW(C.byref(msg))
        except Exception as exc:
            self.error=exc;self.ready.set()
