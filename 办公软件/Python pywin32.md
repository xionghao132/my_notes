# Python pywin32 

## 一、安装

- 是一个针对Windows平台对Python做的扩展
- 包装了Windows 系统的 Win32 API，能创建和使用 COM 对象和图形窗口界面

```python
pip install pywin32
```

## 二、通过标题获取窗口句柄

- 通过标题查找，仅返回一个顶层窗口的句柄
- 不支持模糊查询

```python
import win32gui

# 获取窗口句柄
handle = win32gui.FindWindow(None, '窗口名字')  
# 返还窗口信息（x,y坐标，还有宽度，高度）
handleDetail = win32gui.GetWindowRect(handle)
```

## 三、通过坐标获取窗口句柄

```python
import win32gui

hid = win32gui.WindowFromPoint((100, 100))
```

## 四、通过句柄获取窗口信息

```python
import win32gui

hid = win32gui.WindowFromPoint((100, 100))
# 获取窗口标题
title = win32gui.GetWindowText(hid)
# 获取窗口类名
class_name = win32gui.GetClassName(hid)
```

## 五、通过句柄设置窗口位置大小

```python
import win32gui

hid = win32gui.WindowFromPoint((100, 100))

# 参数：句柄，窗口左边界，窗口上边界，窗口宽度，窗口高度，确定窗口是否被刷新
win32gui.MoveWindow(hid, 100, 100, 800, 800, True)
```

## 六、激活句柄窗口

- 激活指定句柄的窗口

```python
import win32gui

hid = win32gui.WindowFromPoint((100, 100))

# 将创建指定窗口的线程设置到前台，并且激活该窗口
win32gui.SetForegroundWindow(hid)
```

## 七、鼠标位置的设置和获取

```python
import win32api

# 设置位置
win32api.SetCursorPos((100, 100))
# 获取位置
point = win32api.GetCursorPos()
print(point)
```

## 八、鼠标点击事件

- 可以通过 win32api.mouse_event(flags, x, y, data, extra_info) 进行鼠标操作
- MOUSEEVENTF_LEFTDOWN：表明接按下鼠标左键
- MOUSEEVENTF_LEFTUP：表明松开鼠标左键
- MOUSEEVENTF_RIGHTDOWN：表明按下鼠标右键
- MOUSEEVENTF_RIGHTUP：表明松开鼠标右键
- MOUSEEVENTF_MIDDLEDOWN：表明按下鼠标中键
- MOUSEEVENTF_MIDDLEUP：表明松开鼠标中键
- MOUSEEVENTF_WHEEL：鼠标轮移动,数量由data给出

```python
import win32api
import win32con

# 模拟鼠标在(400, 500)位置进行点击操作
point = (400, 500)
win32api.SetCursorPos(point)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
```

## 九、键盘事件

- 通过 keybd_event(bVk, bScan, dwFlags, dwExtraInfo) 可以进行监听键盘事件
- bVk：虚拟键码
- bScan：硬件扫描码，一般设置为0即可
- dwFlags：函数操作的一个标志位，如果值为KEYEVENTF_EXTENDEDKEY则该键被按下，也可设置为0即可，如果值为KEYEVENTF_KEYUP则该按键被释放
- dwExtraInfo：定义与击键相关的附加的32位值，一般设置为0即可

```python
import win32api
import win32con

# 按下ctrl+s
win32api.keybd_event(0x11, 0, 0, 0)
win32api.keybd_event(0x53, 0, 0, 0)
win32api.keybd_event(0x53, 0, win32con.KEYEVENTF_KEYUP, 0)
win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)
```

[python 包之 pywin32 操控 windows 系统教程 - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1987807)

