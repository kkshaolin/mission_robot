import time

dist_tof = None

def read_sensor_IRandTof(ep_sensor):
    def sub_data_handler(sub_info):
        global dist_tof
        distance = sub_info
        dist_tof = distance[0]  # update value
    
    # subscribe (returns True/False, not the distance!)
    ep_sensor.sub_distance(freq=20, callback=sub_data_handler) 
    
    return dist_tof

def find_wall():
    #forward

    #right

    #left

    return 