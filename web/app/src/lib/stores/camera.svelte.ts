export type CameraState = 'off' | 'on' | 'error';

export const camera = $state<{ state: CameraState }>({ state: 'off' });
export const setCameraState = (state: CameraState) => {
  camera.state = state;
};
