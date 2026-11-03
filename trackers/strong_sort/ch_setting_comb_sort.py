import numpy as np
import math 

def chpre_P2R(ch, preset):
    if ch == '01':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 7.16195827e-03, -9.30480189e-02,  3.72907348e+01],
                    [ 2.44124203e-02, -3.17157327e-01,  1.27107903e+02],
                    [ 1.92066107e-04, -2.49523160e-03,  1.00000000e+00]]
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

                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 5.33782008e-03, -8.01632463e-02,  3.72906208e+01],
 [ 1.81949001e-02, -2.73239173e-01,  1.27106470e+02],
 [ 1.43147953e-04, -2.14970482e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '02':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-2.51835128e-03, -8.07184827e-02,  3.72900673e+01],
 [-8.58470581e-03, -2.75131557e-01 , 1.27104838e+02],
 [-6.75395531e-05, -2.16459479e-03 , 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.05949929e-02,  3.96301984e-01,  3.72929799e+01],
 [-3.61100047e-02,  1.35080888e+00,  1.27119731e+02],
 [-2.84090355e-04,  1.06274656e-02,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 3:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 7.11518375e-03, -6.31686669e-01,  3.72851314e+01],
 [ 2.42484778e-02, -2.15313359e+00,  1.27078149e+02],
 [ 1.90768564e-04, -1.69396284e-02,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '03':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 4.89888916e-05, -4.57401893e-02,  3.72917020e+01],
 [ 1.67275791e-04, -1.55911506e-01,  1.27113905e+02],
 [ 1.31601228e-06, -1.22655647e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-2.54609906e-03, -9.17294086e-02,  3.72918552e+01],
 [-8.67783562e-03 ,-3.12672117e-01,  1.27113947e+02],
 [-6.82692974e-05, -2.45978843e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '04':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 5.81178710e-03, -1.04229260e-01,  3.72913922e+01],
 [ 1.98093366e-02, -3.55279481e-01,  1.27112626e+02],
 [ 1.55839699e-04, -2.79498133e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 5.95638040e-03, -1.36132741e-01,  3.72911852e+01],
 [ 2.03018751e-02, -4.64026978e-01,  1.27110272e+02],
 [ 1.59714920e-04, -3.65049572e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '05':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-3.34594137e-03, -5.40298372e-02, 3.72909042e+01],
 [-1.14050924e-02, -1.84180767e-01,  1.27125245e+02],
 [-8.97188034e-05, -1.44886094e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.69045127e-03, -3.53429505e-02 , 3.72913125e+01],
 [-5.76217423e-03, -1.20478827e-01,  1.27121235e+02],
 [-4.53284741e-05, -9.47752475e-04 , 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 3:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-7.74288896e-04, -7.73950136e-02,  3.72914409e+01],
 [-2.63871986e-03, -2.63827955e-01,  1.27121075e+02],
 [-2.07576105e-05, -2.07541553e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '06':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-8.55962441e-03, -9.64931518e-02 , 3.72913542e+01],
 [-2.91794223e-02, -3.28930455e-01 , 1.27119816e+02],
 [-2.29541616e-04, -2.58754907e-03 , 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.78295907e-04, -9.31562035e-02,  3.72915864e+01],
 [-6.08202329e-04, -3.17555709e-01 , 1.27118582e+02],
 [-4.78639348e-06, -2.49806662e-03, 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '07':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.74722614e-01 , 1.77912914e+00  ,3.73070480e+01],
 [-5.95658711e-01 , 6.06536489e+00  ,1.26992452e+02],
 [-4.68558804e-03,  4.77104992e-02  ,1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 2.21746359e-03, -3.93840172e-02,  3.72900704e+01],
 [ 7.55994707e-03, -1.34266619e-01  ,1.27129149e+02],
 [ 5.94677420e-05, -1.05615119e-03 , 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag

        elif preset == 3:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 4.62949949e-04, -5.53346555e-02,  3.72901881e+01],
 [ 1.57873712e-03, -1.88644897e-01,  1.27128616e+02],
 [ 1.24184947e-05, -1.48389504e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '08':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.43499989e-02, -2.78610091e-01,  3.72901038e+01],
 [-4.89238495e-02, -9.49827230e-01,  1.27126284e+02],
 [-3.84840343e-04, -7.47141401e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-6.09662026e-03, -7.33308704e-02 , 3.72903717e+01],
 [-2.07852085e-02, -2.49997029e-01 , 1.27126199e+02],
 [-1.63497307e-04, -1.96649537e-03 , 1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '09':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 2.63939211e-03, -4.70603229e-02 , 3.72891754e+01],
 [ 8.99896216e-03, -1.60446999e-01 , 1.27134851e+02],
 [ 7.07842975e-05, -1.26203364e-03  ,1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 1.76799581e-03, -1.00048086e-01,  3.72893860e+01],
 [ 6.02864549e-03, -3.41102857e-01,  1.27134607e+02],
 [ 4.74198351e-05, -2.68302421e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    elif ch == '10':
        if preset == 1:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[ 4.18111684e-04, -9.83778158e-02,  3.72892954e+01],
                    [ 1.42489007e-03, -3.35408167e-01,  1.27133049e+02],
                    [ 1.12068471e-05, -2.63823214e-03,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
        elif preset == 2:
            def convPtoR(x1,y1,x2,y2) :
                minus_flag = False
                H = [[-1.41488898e-03, -3.27523838e-02,  3.72893267e+01],
 [-4.82417146e-03, -1.11665765e-01,  1.27133240e+02],
 [-3.79456475e-05, -8.78332846e-04,  1.00000000e+00]]
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
                R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)
                return R_dist, minus_flag
    
    return convPtoR

def ch_line(ch,preset):

    up_line = [(355, 756), (1078, 785)]
    down_line = [(0, 1347), (1077, 1333)]
    vup_line = [(354, 784), (1072, 816)]
    vdown_line = [(0, 1267), (1079, 1266)]
    mask = np.array([[0,0], [1, 805], [595, 600], [680,540], [1080, 523], [1080, 6]], np.int32)
    lane_1 = np.array([[297,208],	[289,211],	[282,213],	[276,215],	[270,216],	[267,218],	[261,221],	[255,223],	[251,225],	[246,228],	[241,230],	[234,232],	[227,235],	[223,236],	[216,240],	[212,241],	[206,244],	[200,246],	[195,248],	[187,252],	[180,255],	[177,256],	[170,260],	[163,267],	[159,272],	[156,279],	[153,285],	[152,290],	[151,292],	[149,304],	[149,314],	[149,329],	[150,338],	[153,370],	[154,386],	[155,407],	[160,420],	[170,433],	[185,449],	[199,463],	[219,482],	[241,507],	[265,530],	[274,539],	[291,557],	[320,585],	[441,703],	[491,750],	[553,809],	[720,967],	[720,757],	[704,746],	[676,725],	[658,713],	[638,697],	[618,684],	[602,671],	[586,661],	[568,646],	[554,636],	[537,625],	[523,615],	[513,606],	[386,509],	[367,494],	[360,488],	[344,477],	[337,469],	[326,460],	[318,453],	[309,445],	[302,439],	[292,431],	[287,425],	[281,421],	[277,417],	[272,412],	[268,408],	[263,404],	[259,400],	[256,397],	[252,393],	[247,386],	[240,377],	[238,371],	[235,368],	[230,358],	[223,348],	[218,338],	[215,331],	[211,323],	[208,316],	[207,308],	[203,299],	[203,292],	[202,284],	[202,279],	[203,272],	[203,267],	[208,259],	[221,251],	[237,244],	[247,240],	[262,232],	[272,227],	[284,220],	[295,215],	[305,212]], np.int32)
    lane_2 = np.array([[245,216],	[231,223],	[218,228],	[204,233],	[190,237],	[172,244],	[159,248],	[152,251],	[141,255],	[131,259],	[121,266],	[109,279],	[100,292],	[92,301],	[88,311],	[84,322],	[79,334],	[78,339],	[74,358],	[69,371],	[64,386],	[61,417],	[62,436],	[65,443],	[72,456],	[83,476],	[93,498],	[108,525],	[117,540],	[122,549],	[132,572],	[146,601],	[215,729],	[289,847],	[533,1207],	[586,1280],	[720,1280],	[720,968],	[551,807],	[521,783],	[489,755],	[435,704],	[387,656],	[316,588],	[269,539],	[211,477],	[170,436],	[158,421],	[154,409],	[148,335],	[147,320],	[147,305],	[148,291],	[153,279],	[159,267],	[176,254],	[185,250],	[195,246],	[206,241],	[230,232],	[250,224]], np.int32)
    lane_3 = np.array([[238,213],	[226,217],	[204,223],	[171,233],	[133,244],	[95,257],	[75,264],	[52,272],	[45,277],	[41,286],	[37,299],	[0,351],	[0,901],	[88,1280],	[571,1280],	[526,1211],	[353,947],	[211,731],	[153,619],	[117,547],	[104,524],	[59,435],	[62,383],	[79,329],	[83,313],	[97,290],	[104,282],	[112,272],	[123,263],	[132,258],	[145,251],	[173,241],	[190,236],	[230,222],	[244,215]], np.int32)
 
 
    return up_line, down_line, vup_line, vdown_line, mask, lane_1, lane_2, lane_3