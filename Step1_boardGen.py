import sleap_anipose as slap

def slap_draw_board(rootpath):
    slap.draw_board(board_name = f"{rootpath}/board.jpg", 
                    board_x = 8, 
                    board_y = 11, 
                    square_length = 24.0, 
                    marker_length = 19.75, 
                    marker_bits = 4, 
                    dict_size = 1000, 
                    img_width = 1440, 
                    img_height = 1440, 
                    save = f"{rootpath}/board.toml")
    
if __name__ == "__main__":
    rootpath = "D:/Project/Sleap-Model/3dT"
    slap_draw_board(rootpath)