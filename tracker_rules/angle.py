""" Module defines basic track classification rules based on angle rules """

from openem.tracking import track_vel

def classify_track(media_id,
                   proposed_track_element,
                   minimum_length=2,
                   label='Label',
                   names={}):
    
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
        Example:
        # Sets 2 classes, entering (Left quadrants) exit (+/- 45 over right)
        # minimum track length is 20 detections
        strategy_args = {"minimum_length": 20,
                         "label": "Direction",
                         "names": {
                                          #Low,High
                             "Entering": [-45,45],
                             "Exiting": [90,180],
                             "Unknown": [-180,180] # catch all
                                  }
                         }
    """
    if len(proposed_track_element) >= minimum_length:
        angle,speed,_ = track_vel(proposed_track_element)
        for class_name,angles in names:
            if angle >= angles[0] and angle <= angles[1]:
                return True,{label:class_name,
                             'length': len(proposed_track_element),
                             'speed': speed,
                             'angle': angle}
    return False,None
