""" Module defines basic track classification rules based on angle rules """

import math

def track_vel(track):
    """ calculates bearing and velocity of a track [first to last] """
    track.sort(key=lambda x:x['frame'])
    track_len = track[-1]['frame'] - track[0]['frame']

    if 'orig_x' in track[0]:
        f_cx = track[0]['orig_x'] + (track[0]['orig_w']/2)
        f_cy = track[0]['orig_y'] + (track[0]['orig_h']/2)
    else:
        f_cx = track[0]['x'] + (track[0]['width']/2)
        f_cy = track[0]['y'] + (track[0]['height']/2)

    if 'orig_x' in track[-1]:
        l_cx = track[-1]['orig_x'] + (track[-1]['orig_w']/2)
        l_cy = track[-1]['orig_y'] + (track[-1]['orig_h']/2)
    else:
        l_cx = track[-1]['x'] + (track[-1]['width']/2)
        l_cy = track[-1]['y'] + (track[-1]['height']/2)

    print(f"{track[0]['frame']}: {f_cx},{f_cy} to {l_cx,l_cy}")
    x_vel = (l_cx-f_cx)  / track_len
    y_vel = (l_cy - f_cy) / track_len
    magnitude=math.sqrt(math.pow(x_vel,2)+math.pow(y_vel,2))
    if magnitude <= 0.00001:
        magnitude = 0
        angle = 0
        x_vel = 0
        y_vel = 0
        distance = 0
    else:
        angle = math.atan2(y_vel, x_vel)
        # unfurl radian
        if angle < 0:
            angle = 2*math.pi + angle
        distance = np.sqrt(math.pow(l_cx-f_cx, 2) + math.pow(l_cy-f_cy, 2))
    return (angle, magnitude,[x_vel,y_vel], distance)

def classify_track(media_id,
                   proposed_track_element,
                   minimum_length=2,
                   label='Label',
                   names={},
                   minimum_distance=0):
    
    """ 
        :media_id: ID of media track will belong to
        :proposed_track_element: list of detections that belong to 
                                 the proposed track 
        :returns tuple: where first element is boolean on whether the proposed
        track is valid, and the 2nd are the attributes to apply.

        - minimum_length : number of detections to consider valid
                              (Default: 2)
        - label: "Name to use for the direction assignment" (default label)
        - names : Dictionary of names to angle space (inclusive)
                              (Default: {})
        - minimum_distance: Minimum number of pixels traveled by the track (last minus first).
        Example:
        # Sets 2 classes, entering (Left quadrants) exit (+/- 45 over right)
        # minimum track length is 20 detections
        strategy_args = {"minimum_length": 20,
                         "label": "Direction",
                         "names": {
                                          #Low,High
                             "Entering": [[50,85],[315,360]]
                             "Exiting": [[90,270]],
                             "Unknown": [[0,360]] # catch all
                                  }
                         }
    """
    if len(proposed_track_element) >= minimum_length:
        angle,speed,_, distance = track_vel(proposed_track_element)
        if distance >= minimum_distance:
            angle = math.degrees(angle)
            for class_name,angles_list in names.items():
                for angles in angles_list:
                    if angle >= angles[0] and angle <= angles[1]:
                        return True,{label:class_name,
                                     'length': len(proposed_track_element),
                                     'speed': speed,
                                     'angle': angle,
                                     'distance': distance}
    return False,None
