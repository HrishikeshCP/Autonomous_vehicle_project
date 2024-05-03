from rplidar import RPLidar

def run(PORT_NAME,revs=1):

    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    lidar.reset()
    history={}
    ctr=0
    try:
        # print('Recording measurments... Press Crl+C to stop.')
        for measurment in lidar.iter_measurments():
        #     line = '\t'.join(str(v) for v in measurment)
        #     outfile.write(line + '\n')
            boolean, quality, angle, distance = measurment  # Unpack values here
            if boolean:
                ctr+=1
                history={}
            if ctr==revs:
                break
            if distance!=0.0 and (300 < int(angle) or 30>int(angle)):
                history[angle]=distance
            if len(history)>0:
                min_key = min(history.values())  
            
    except KeyboardInterrupt:
        print('')
    lidar.stop()
    lidar.disconnect()
    return min_key
    

# if __name__ == '__main__':
    # run()