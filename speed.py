ms_to_kmh = 3.6
fps = 30

def sum_ssl(speed_stack_length):
    total_sum = 0
    for i in range(int(speed_stack_length / 2)):
        total_sum += (i + 1)
    return total_sum

def CalcAverageSpeed(ms_to_kmh, track, convP2R, channel):
    speed_sum = 0
    
    for j in range(int(len(track.matching_point_list) / 2)):
        
        x1 = track.matching_point_list[j][0]
        y1 = track.matching_point_list[j][1]
        x2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][0]
        y2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][1]
        r_dist, minus_flag = convP2R(x1, y1, x2, y2)

        time1 = track.matching_point_list[j][2]
        time2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][2]
        # t_time = (time2 - time1) / 1000
        t_time = (time2 - time1) / fps
        speed = r_dist / t_time * ms_to_kmh
        if minus_flag:
            speed *= -1
        if channel == '01' or channel == '03' or channel == '05' or channel == '07' or channel == '09':
            speed = (int(len(track.matching_point_list) / 2)-j) / sum_ssl(len(track.matching_point_list)) * speed
        else:
            speed = (j + 1) / sum_ssl(len(track.matching_point_list)) * speed
        speed_sum += speed

    return int(speed_sum)

def CarcCurrentSpeed(ms_to_kmh, track, convP2R, current_list):
    x1 = track.matching_point_list[0][0]
    y1 = track.matching_point_list[0][1]
    x2 = current_list[0]
    y2 = current_list[1]
    r_dist, minus_flag = convP2R(x1, y1, x2, y2)
    
    time1 = track.matching_point_list[0][2]
    time2 = current_list[2]
    # t_time = (time2 - time1) / 1000
    t_time = (time2 - time1) / 30.0
    speed = r_dist / t_time * ms_to_kmh
    if minus_flag:
        speed *= -1
    return int(speed)

def CalcSpeed(track, convP2R, current_list, channel, avg_speed_list):
    sspeed = 0
    avg_speed = 0
    speed_stack_length = 20
    
    # print("track id",track.track_id, "length", len(track.matching_point_list))
    if len(track.matching_point_list) < speed_stack_length:
        track.pop_flag = False
        track.speed_valid = False
    else:
        track.pop_flag = True
        track.speed_valid = True
    track.matching_point_list.append(current_list)
        
    if track.speed_valid:
        if avg_speed_list[track.driving_lane-1] == -999: # if lane avrage don't exists
            sspeed = CarcCurrentSpeed(ms_to_kmh, track, convP2R, current_list)
            if sspeed >= 250:
                track.matching_point_list.pop()
                track.pop_flag = False
        else: # if lane avrage exists
            avg_lane_speed = avg_speed_list[track.driving_lane-1]
            avg_speed = CalcAverageSpeed(ms_to_kmh, track, convP2R, channel)
            sspeed = CarcCurrentSpeed(ms_to_kmh, track, convP2R, current_list)
            if avg_speed > 0:                       
                if sspeed != 0:
                    if sspeed > 0:
                        if sspeed >= 250: # if current speed exceed 250km/h, delete this point
                            track.matching_point_list.pop()
                            track.pop_flag = False
                            
                        elif 0 < sspeed <= 15:
                            if avg_speed > 0:
                                # if current speed +-10%, car average speed replaces current speed
                                if sspeed < 0.90 * avg_speed or sspeed > 1.10 * avg_speed:
                                    sspeed = avg_speed
                                    track.matching_point_list.pop()
                                    track.pop_flag = False
                        else:
                            # if current speed +-10%, car average speed replaces current speed
                            if sspeed < 0.90 * avg_speed or sspeed > 1.10 * avg_speed:
                                # if current speed +-10% and current speed +-30%, lane average speed replaces current speed
                                if sspeed < 0.70 * avg_lane_speed or sspeed > 1.30 * avg_lane_speed: 
                                    # if lane average speed is almost close to zero, manipulation stop
                                    if 0 <= avg_lane_speed <= 20:
                                        if sspeed > 4 * avg_lane_speed:
                                            sspeed = avg_lane_speed
                                            track.matching_point_list.pop()
                                            track.pop_flag = False  
                                    else:
                                        sspeed = avg_lane_speed
                                        track.matching_point_list.pop()
                                        track.pop_flag = False   
                                else:
                                    if -20 <= avg_speed <= 20:
                                        pass
                                    else:
                                        sspeed = avg_speed
                                        track.matching_point_list.pop()
                                        track.pop_flag = False
                    elif sspeed < 0:
                        if avg_lane_speed > 0:
                            if sspeed < 0.80 * avg_lane_speed or sspeed > 1.20 * avg_lane_speed:
                                sspeed = avg_lane_speed    
                                track.matching_point_list.pop()
                                track.pop_flag = False
            else:
                if avg_lane_speed > 0:
                    if sspeed < 0.80 * avg_lane_speed or sspeed > 1.20 * avg_lane_speed:
                        sspeed = avg_lane_speed    
                        track.matching_point_list.pop()
                        track.pop_flag = False

        if track.pop_flag:
            track.matching_point_list.pop(0)

        return int(sspeed)