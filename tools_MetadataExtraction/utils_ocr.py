
def adapt_position(position, resize_factor):
    position_adapeted = []
    for i_position in position:
        i_position = [int(i/resize_factor) for i in i_position]
        position_adapeted.append(i_position)

    return position_adapeted
