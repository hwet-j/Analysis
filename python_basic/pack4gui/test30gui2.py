# 레이아웃 관리자 클래스 중 BoxSizer 연습
import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title = title, size=(300, 350))
        
        panel1 = wx.Panel(self, -1, style = wx.SUNKEN_BORDER)
        panel2 = wx.Panel(self, -1, style = wx.SUNKEN_BORDER)
        
        panel1.SetBackgroundColour('BLUE')
        panel2.SetBackgroundColour('red')
        
        # 현재 공부하는 내용에서 이쪽 부분을 중점을 두고 볼것.
        box = wx.BoxSizer(wx.VERTICAL)  # VERTICAL, HORIZONTAL
        box.Add(panel1, 1, wx.EXPAND)
        box.Add(panel2, 1, wx.EXPAND)
        
        self.SetSizer(box)
        self.Center()
        self.Show()
        
if __name__ == '__main__':
    app = wx.App()
    MyFrame(None, title = '레이아웃')
    app.MainLoop()