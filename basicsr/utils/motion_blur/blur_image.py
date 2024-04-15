import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
# from scipy import misc
# from imageio import imread
import torch
import kornia
from basicsr.utils.motion_blur.generate_PSF import PSF
from basicsr.utils.motion_blur.generate_trajectory import Trajectory
# import PySide2
#
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
class BlurGrayImage(object):

    def __init__(self,  PSFs=None, part=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_gray_image(self, image):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        # yN, xN = image.shape[:2]
        # key, kex = self.PSFs[0].shape
        # delta = yN - key
        # assert delta >= 0, 'resolution of image should be higher than kernel'
        if len(psf) > 1:
            for p in psf:
                p_k = np.random.uniform(0., 1.)
                # print('p: ', p.shape)
                if p_k < 0.5:
                    idx = np.where(p > 1e-5)
                    p[idx] = np.exp(p[idx])
                    p[idx] = p[idx] / np.sum(p[idx])
                else:
                    p = np.abs(p)
                    p = p / np.sum(p)
                # tmp = np.pad(p, delta // 2, 'constant')
                # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                # blured = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                #                        dtype=cv2.CV_32F)
                tmp_x = kornia.image_to_tensor(p, keepdim=False)
                tmp_x = tmp_x.squeeze(0).to(image.device)
                blured = kornia.filters.filter2d(image, tmp_x)
                # blured = np.array(signal.fftconvolve(image, tmp, 'same'))
                # blured = cv2.normalize(blured, blured, alpha=image.min(), beta=image.max(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            psf = psf[0]
            # print('psf: ', psf.shape, psf.min(), len(np.where(psf<1e-1)))
            # print(psf)
            p_k = np.random.uniform(0., 1.)
            # print('p: ', p.shape)
            if p_k < 0.5:
                idx = np.where(psf>1e-5)
                psf[idx] = np.exp(psf[idx])
                psf[idx] = psf[idx] / np.sum(psf[idx])
            else:
                psf = np.abs(psf)
                psf = psf / np.sum(psf)
            # tmp = np.pad(psf, delta // 2, 'constant')
            # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                        dtype=cv2.CV_32F)
            tmp_x = kornia.image_to_tensor(psf, keepdim=False)
            tmp_x = tmp_x.squeeze(0).to(image.device)
            blured = kornia.filters.filter2d(image, tmp_x)
            # blured = np.array(signal.fftconvolve(image, tmp, 'same'))
            # blured = cv2.normalize(blured, blured, alpha=image.min(), beta=image.max(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return blured # np.clip(blured, 0., 1.)

class BlurImage(object):

    def __init__(self, image_path=None, PSFs=None, part=None, path__to_save=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        # if os.path.isfile(image_path):
        #     self.image_path = image_path
        #     self.original = cv2.imread(self.image_path) / 255.
        #     self.shape = self.original.shape
        #     if len(self.shape) < 3:
        #         raise Exception('We support only RGB images yet.')
        #     elif self.shape[0] != self.shape[1]:
        #         raise Exception('We support only square images yet.')
        # else:
        #     raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                p_k = np.random.uniform(0., 1.)
                # print('p: ', p.shape)
                if p_k < 0.5:
                    idx = np.where(p > 1e-5)
                    p[idx] = np.exp(p[idx])
                    p[idx] = p[idx] / np.sum(p[idx])
                else:
                    p = np.abs(p)
                    p = p / np.sum(p)
                # tmp = np.pad(p, delta // 2, 'constant')
                # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                # blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                #                        dtype=cv2.CV_32F)
                # blured = self.original
                # blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                # blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                # blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured_ = kornia.image_to_tensor(self.original, keepdim=False)
                tmp_x = kornia.image_to_tensor(p, keepdim=False)
                tmp_x = tmp_x.squeeze(0)
                blured_ = kornia.filters.filter2d(blured_, tmp_x)
                blured_ = kornia.tensor_to_image(blured_)
                # blured_np = np.fft.rfft2(blured, axes=[0, 1]) * np.fft.rfft2(tmp, axes=[0, 1])
                # print('xxx: ', np.mean(blured - blured_))
                # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.clip(blured_, 0., 1.))
        else:
            psf = psf[0]
            # print('psf: ', psf.shape, psf.min(), len(np.where(psf<1e-1)))
            # print(psf)
            p_k = np.random.uniform(0., 1.)
            # print('p: ', p.shape)
            if p_k < 0.5:
                idx = np.where(psf>1e-5)
                psf[idx] = np.exp(psf[idx])
                psf[idx] = psf[idx] / np.sum(psf[idx])
            else:
                psf = np.abs(psf)
                psf = psf / np.sum(psf)
            # tmp = np.pad(psf, delta // 2, 'constant')
            # print(tmp.shape)
            ke = max(key, kex)
            ke = ke // 2
            # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                        dtype=cv2.CV_32F)
            # blured = self.original
            # blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            # blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            # blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))

            blured_ = kornia.image_to_tensor(self.original, keepdim=False)
            tmp_x = kornia.image_to_tensor(psf, keepdim=False)
            tmp_x = tmp_x.squeeze(0)

            blured_ = kornia.filters.filter2d(blured_.cuda(), tmp_x.cuda())
            blured_ = kornia.tensor_to_image(blured_.cpu())

            # blured_ = kornia.image_to_tensor(self.original, keepdim=False)
            # tmp_x = kornia.image_to_tensor(psf, keepdim=False)
            # # tmp_x = tmp_x.squeeze(0)
            # pad_1 = delta // 2
            # pad_2 = delta // 2 if delta % 2 == 0 else delta // 2 + 1
            # tmp_x = torch.nn.functional.pad(tmp_x, (ke+pad_1, ke+pad_2, ke+pad_1, ke+pad_2), mode='constant')
            # blured_ = torch.nn.functional.pad(blured_, (ke, ke, ke, ke), mode='reflect')
            # # print(blured_.shape, tmp_x.shape)
            # blured_ = torch.fft.irfft2(torch.fft.rfft2(blured_) * torch.conj(torch.fft.rfft2(tmp_x)))
            # blured_ = torch.fft.fftshift(blured_, dim=[-1,-2])
            # blured_ = kornia.tensor_to_image(blured_.cpu())

            # print('xxx: ', np.mean(blured - blured_))
            # blured_np = self.original
            # print(psf.shape)
            #
            # tmp = np.pad(tmp, ke, 'constant')
            # tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)
            # blured_np = np.pad(blured_np, ((ke, ke), (ke, ke), (0, 0)), 'reflect')
            # print(blured_np.shape, tmp.shape)
            # blured_np = np.fft.irfft2(np.fft.rfft2(blured_np, axes=[0, 1]) * np.conj(np.fft.rfft2(tmp, axes=[0, 1])), axes=[0,1])
            # blured_np = np.fft.fftshift(blured_np, axes=[0, 1])
            # print('yyy: ', np.mean(blured - blured_np))
            # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.clip(blured_[ke:yN+ke, ke:xN+ke, :], 0., 1.)) # [1,2,0]
        self.result = result
        if show or save:
            self.__plot_canvas(show, save)
        return blured_
    def blur_image_(self, original, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = original.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                p_k = np.random.uniform(0., 1.)
                # print('p: ', p.shape)
                if p_k < 0.5:
                    idx = np.where(p > 1e-5)
                    p[idx] = np.exp(p[idx])
                    p[idx] = p[idx] / np.sum(p[idx])
                else:
                    p = np.abs(p)
                    p = p / np.sum(p)
                # tmp = np.pad(p, delta // 2, 'constant')
                # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                # blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                #                        dtype=cv2.CV_32F)
                # blured = self.original
                # blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                # blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                # blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured_ = kornia.image_to_tensor(original, keepdim=False)
                tmp_x = kornia.image_to_tensor(p, keepdim=False)
                tmp_x = tmp_x.squeeze(0)
                blured_ = kornia.filters.filter2d(blured_, tmp_x)
                blured_ = kornia.tensor_to_image(blured_)
                # blured_np = np.fft.rfft2(blured, axes=[0, 1]) * np.fft.rfft2(tmp, axes=[0, 1])
                # print('xxx: ', np.mean(blured - blured_))
                # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.clip(blured_, 0., 1.))
        else:
            psf = psf[0]
            # print('psf: ', psf.shape, psf.min(), len(np.where(psf<1e-1)))
            # print(psf)
            p_k = np.random.uniform(0., 1.)
            # print('p: ', p.shape)
            if p_k < 0.5:
                idx = np.where(psf>1e-5)
                psf[idx] = np.exp(psf[idx])
                psf[idx] = psf[idx] / np.sum(psf[idx])
            else:
                psf = np.abs(psf)
                psf = psf / np.sum(psf)
            # tmp = np.pad(psf, delta // 2, 'constant')
            # print(tmp.shape)
            ke = max(key, kex)
            ke = ke // 2
            # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                        dtype=cv2.CV_32F)
            # blured = self.original
            # blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            # blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            # blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))

            blured_ = kornia.image_to_tensor(original, keepdim=False)
            tmp_x = kornia.image_to_tensor(psf, keepdim=False)
            tmp_x = tmp_x.squeeze(0)

            blured_ = kornia.filters.filter2d(blured_.cuda(), tmp_x.cuda())
            blured_ = kornia.tensor_to_image(blured_.cpu())

            # blured_ = kornia.image_to_tensor(self.original, keepdim=False)
            # tmp_x = kornia.image_to_tensor(psf, keepdim=False)
            # # tmp_x = tmp_x.squeeze(0)
            # pad_1 = delta // 2
            # pad_2 = delta // 2 if delta % 2 == 0 else delta // 2 + 1
            # tmp_x = torch.nn.functional.pad(tmp_x, (ke+pad_1, ke+pad_2, ke+pad_1, ke+pad_2), mode='constant')
            # blured_ = torch.nn.functional.pad(blured_, (ke, ke, ke, ke), mode='reflect')
            # # print(blured_.shape, tmp_x.shape)
            # blured_ = torch.fft.irfft2(torch.fft.rfft2(blured_) * torch.conj(torch.fft.rfft2(tmp_x)))
            # blured_ = torch.fft.fftshift(blured_, dim=[-1,-2])
            # blured_ = kornia.tensor_to_image(blured_.cpu())

            # print('xxx: ', np.mean(blured - blured_))
            # blured_np = self.original
            # print(psf.shape)
            #
            # tmp = np.pad(tmp, ke, 'constant')
            # tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)
            # blured_np = np.pad(blured_np, ((ke, ke), (ke, ke), (0, 0)), 'reflect')
            # print(blured_np.shape, tmp.shape)
            # blured_np = np.fft.irfft2(np.fft.rfft2(blured_np, axes=[0, 1]) * np.conj(np.fft.rfft2(tmp, axes=[0, 1])), axes=[0,1])
            # blured_np = np.fft.fftshift(blured_np, axes=[0, 1])
            # print('yyy: ', np.mean(blured - blured_np))
            # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.clip(blured_[ke:yN+ke, ke:xN+ke, :], 0., 1.)) # [1,2,0]
        self.result = result
        if show or save:
            self.__plot_canvas(show, save)
        return blured_
    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
            elif show:
                plt.show()


if __name__ == '__main__':
    import tqdm
    folder = '/media/mxt/9a254338-2f19-427f-8f68-6d041ef327d9/Dataset/GoPro/val/target_crops'
    folder_to_save = '/media/mxt/9a254338-2f19-427f-8f68-6d041ef327d9/Dataset/GoPro/val/target_crops_reblur_fft'
    os.makedirs(folder_to_save, exist_ok=True)
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    for path in tqdm.tqdm(os.listdir(folder)):
        # print(path)
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        BlurImage(os.path.join(folder, path), PSFs=psf,
                  path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
            blur_image(save=True)
