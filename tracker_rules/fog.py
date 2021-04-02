def total_movement(boxes):
  import math
  this_box = boxes[0]
  next_box = boxes[-1]
  x0 = (this_box.x + this_box.width)/2
  y0 = (this_box.y + this_box.height)/2
  x1 = (next_box.x + next_box.width)/2
  y1 = (next_box.y + next_box.height)/2
  return math.sqrt(math.pow((x1-x0),2)+math.pow((y1-y0),2))

def is_fog(api, project, track_id, thumbnails, threshold=15, movement_threshold=0.4):
  import cv2
  track_obj = api.get_state(track_id)
  localizations = api.get_localization_list_by_id(project, {"ids": track_obj.localizations})
  localizations.sort(key=lambda x:x.frame)
  movement = total_movement(localizations)
  label,track_entropy=None,None
  grayscales = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in thumbnails]
  lum_mean_sum = 0.0
  lum_std_sum = 0.0
  for gs in grayscales:
    lum_mean_sum += gs.mean()
    lum_std_sum += gs.std()
  lum_mean = lum_mean_sum / len(grayscales)
  lum_std = lum_std_sum / len(grayscales)
  if lum_std <= threshold and movement < movement_threshold:
    label = 'Fog'
    track_entropy = 0
  return label,track_entropy
