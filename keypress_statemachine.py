from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    import tkinter as tk
else:
    import Tkinter as tk


class Application(tk.Tk):

    def __init__(self):
        super().__init__()
        self.entry = tk.Entry(self, w=25)
        self.entry.pack()

        self.bind("<KeyPress>", self._handle_keydown)
        self.bind("<KeyRelease>", self._handle_keyup)

    def _handle_keydown(self, event):
        print(event.char, "down")

    def _handle_keyup(self, event):
        print(event.char, "up")


if __name__ == '__main__':
    app = Application()
    app.mainloop()
