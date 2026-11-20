"""Four mouse controls drawn in the camera image; HighGUI maps clicks to image pixels."""
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MouseControls:
    actions = ("a","d","s","w")
    labels = ("左变道","右变道","−10 km/h","+10 km/h")

    def __init__(self):
        self.pending = deque()
        self.bounds = {}

    @staticmethod
    @lru_cache(maxsize=4)
    def buttons(width,height):
        scale = width/1280
        bw,bh,gap = round(110*scale),round(48*scale),round(8*scale)
        bar = Image.new("RGB",(4*bw+3*gap,bh),(31,34,38))
        painter = ImageDraw.Draw(bar)
        font = ImageFont.truetype(str(Path("C:/Windows/Fonts/msyh.ttc")),round(21*scale))
        for i,label in enumerate(MouseControls.labels):
            x=i*(bw+gap)
            painter.rounded_rectangle((x,0,x+bw-1,bh-1),radius=round(5*scale),fill=(44,58,69))
            painter.text((x+bw/2,bh/2),label,anchor="mm",font=font,fill="white")
        pixels=cv2.cvtColor(np.asarray(bar),cv2.COLOR_RGB2BGR)
        left=(width-bar.width)//2
        top=height-bh-round(39*scale)
        bounds={action:(left+i*(bw+gap),top,bw,bh) for i,action in enumerate(MouseControls.actions)}
        return pixels,left,top,bounds

    def draw(self,view,target,veto=False):
        height,width=view.shape[:2]
        self.image_size=width,height
        pixels,x,y,self.bounds=self.buttons(width,height)
        h,w=pixels.shape[:2]
        view[y:y+h,x:x+w]=pixels
        caption=f"Target {target} km/h" + ("   VETO" if veto else "")
        cv2.putText(view,caption,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,width/2000,(24,24,24),4,cv2.LINE_AA)
        cv2.putText(view,caption,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,width/2000,(255,255,255),1,cv2.LINE_AA)
        return view

    def click(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            for action,(left,top,w,h) in self.bounds.items():
                if left<=x<left+w and top<=y<top+h:
                    self.pending.append(action)
                    break

    def poll(self):
        return self.pending.popleft() if self.pending else ""
