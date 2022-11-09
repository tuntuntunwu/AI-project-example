import pyautogui as pg

print('Press Ctrl+C in terminal to stop the script')
while True:
    x, y = pg.position()
    position_string = "X: " + str(x) + " Y: " + str(y)
    print(position_string, end="")
    print('\b' * len(position_string), end="", flush=True)
