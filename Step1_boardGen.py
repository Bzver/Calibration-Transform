import sleap_anipose as slap

rootpath = 'D:/Project/Sleap-Models/3dT/'

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