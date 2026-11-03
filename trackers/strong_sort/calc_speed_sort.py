ms_to_kmh = 3.6
fps = 30
import geopy

import geopy.distance as geopydist

import numpy as np
def sum_ssl(speed_stack_length):
    total_sum = 0
    for i in range(int(speed_stack_length / 2)):
        total_sum += (i + 1)
    return total_sum

def CalcAverageSpeed(ms_to_kmh, track, convP2R, channel):
    '''
    point) 카메라에 가까운 위치의 픽셀이 호모그래피 상에서 정확하기 때문에 카메라와 차가 다가오는 방향을 고려.
    
    카메라에 가까울 때, 취득되는 속도 값의 신뢰도가 높아 가중치를 크게 줌

    0-5,1-6,2-7,3-8,4-9 프레임들 비교 (개별 가중치 적용)
    '''
    speed_sum = 0
    
    for j in range(int(len(track.matching_point_list) / 2)):
        
        x1 = track.matching_point_list[j][0]
        y1 = track.matching_point_list[j][1]
        x2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][0]
        y2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][1]
        r_dist, minus_flag = convP2R(x1, y1, x2, y2)

        time1 = track.matching_point_list[j][2]
        time2 = track.matching_point_list[int(len(track.matching_point_list) / 2) + j][2]
        # print('time1')
        # print(time1)
        # print('time2')
        # print(time2)
        # t_time = (time2 - time1) / 1000
        t_time = (time2 - time1) / fps
        # print('matching list')
        # print(track.matching_point_list)
        # print('t time')
        # print(t_time)
        # print('r dist')
        # print(r_dist)
        speed = r_dist / t_time * ms_to_kmh
        # print('speed')
        # print(speed)
        if minus_flag:
            speed *= -1
        if channel == '01' or channel == '03' or channel == '05' or channel == '07' or channel == '09':
            #근접한 frame들에 가중치 주기 1/15, 2/15,3/15....
            #역방향/정방향 
            speed = (int(len(track.matching_point_list) / 2)-j) / sum_ssl(len(track.matching_point_list)) * speed
        else:

            speed = (j + 1) / sum_ssl(len(track.matching_point_list)) * speed
        speed_sum += speed
    # print('speed_sum')
    # print(speed_sum)

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

    # print('t_time')
    # print(t_time)
    # print('r_dist')
    # print(r_dist)
    # print('speed')
    # print(speed)
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

                        
# '''
# <자차 평균 속도가 양수일 때> -> 일반적인 상황
# 1. 자차 속도가 저속일 때,
#     1) 자차 평균 속도 보다 +-10%면, 자차 평균속도로 대체
#     2) 자차 평균 속도랑 레인 평균 속도 비교해서 평균속도 대비 +-10% 일때
#         2-1) 레인 평균 속도 대비 +-30% 일 때,
#             - 레인 평균속도가 정체구간(0~20km/h)일 때, 자차 속도가 너무 튀면(*4 초과) 레인 평균 속도로 대체
#             - 레인 평균 속도가 정체구간이 아닐 때, 레인 평균 속도 대체
#         2-2) 레인 평균 속도 대비 +-30% 아닐 때,
#             - 자차 평균 속도가 저속 구간일 때(+-20) 아무 작업 안함
#             - 자차 평균 속도가 저속 구간이 아닐 때, 자차 평균속도로 현재 속도 대체
# 2. 자차 속도가 음수일 때,
#     1) 레인 평균 속도가 양수고, 레인 평균 속도 대비 +-20% 초과일 때, 레인평균 속도를 자차 속도로 대체
# <자차 평균 속도가 음수일 때> -> 역주행 상황
# 1. 레인 평균 속도가 양수일 때, 레인 평균 속도 대비 +-20% 초과일 때, 레인 평균 속도를 자차 속도로 대체
# '''

    if track.speed_valid:
        if avg_speed_list[track.driving_lane-1] == -999: # if lane avrage don't exists
            sspeed = CarcCurrentSpeed(ms_to_kmh, track, convP2R, current_list)
            print('a')
            print(sspeed)
            if sspeed >= 250:
                track.matching_point_list.pop()
                track.pop_flag = False
        else: # if lane avrage exists
            #lane avg speed
            avg_lane_speed = avg_speed_list[track.driving_lane-1]
            #current matching list avg speed
            avg_speed = CalcAverageSpeed(ms_to_kmh, track, convP2R, channel)
            #first frame - latest frame avg speed
            sspeed = CarcCurrentSpeed(ms_to_kmh, track, convP2R, current_list)
            print('b')
            print(sspeed)
            print(avg_speed)
            if avg_speed > 0:                       
                if sspeed != 0:
                    if sspeed > 0:
                        if sspeed >= 250: # if current speed exceed 250km/h, delete this point
                            track.matching_point_list.pop()
                            track.pop_flag = False

                        #정체시 (저속)
                        elif 0 < sspeed <= 15:
                            if avg_speed > 0:
                                # if current speed +-10%, car average speed replaces current speed
                                if sspeed < 0.90 * avg_speed or sspeed > 1.10 * avg_speed:
                                    #matching list avg speed
                                    sspeed = avg_speed
                                    track.matching_point_list.pop()
                                    track.pop_flag = False
                        else:
                            # if current speed +-10%, car average speed replaces current speed
                            if sspeed < 0.90 * avg_speed or sspeed > 1.10 * avg_speed:
                                # if current speed +-10% and current speed +-30%, lane average speed replaces current speed
                                if sspeed < 0.70 * avg_lane_speed or sspeed > 1.30 * avg_lane_speed: 
                                    # if lane average speed is almost close to zero, manipulation stop

                                    #when overall lane is congested
                                    if 0 <= avg_lane_speed <= 20:
                                        #현재 스피드가 lane avg speed *4 보다크면 삭제
                                        #speed에 lane avg speed
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
                    elif sspeed < 0: # 현재 속도가 음수가 나올 때(BBOX 뒤로 밀릴 때)
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




def convPtoR(x1,y1,x2,y2) :
    minus_flag = False
    H = [[ 7.41461933e-03 ,-5.63196412e-01,  1.26611368e+02],
 [ 2.19803547e-03 ,-1.66933111e-01 , 3.75276050e+01],
 [ 5.85638168e-05 ,-4.44821357e-03,  1.00000000e+00]]
    B1 = np.array([x1,y1,1])
    R1 = np.matmul(H,B1)
    B2 = np.array([x2,y2,1])
    R2 = np.matmul(H,B2)
    R1 /= R1[2]
    R2 /= R2[2]
    
    x_distance = ((R1[0]-R2[0]) * 100000 *1.1 ) #보정
    y_distance = ((R1[1]-R2[1]) * 100000 *0.9)
    if y_distance < 0:
        minus_flag = True

    # R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)


    R_dist = geopydist.geodesic((R1[1],R1[0]), (R2[1],R2[0])).meters
    return R_dist, minus_flag

