def is_fog(track_id, thumbnails, threshold=15):
  import cv2
  label,track_entropy=None,None
  grayscales = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in thumbnails]
  lum_mean_sum = 0.0
  lum_std_sum = 0.0
  for gs in grayscales:
    lum_mean_sum += gs.mean()
    lum_std_sum += gs.std()
  lum_mean = lum_mean_sum / len(grayscales)
  lum_std = lum_std_sum / len(grayscales)
  if lum_std <= threshold:
    label = 'Fog'
    track_entropy = 0
  return label,track_entropy