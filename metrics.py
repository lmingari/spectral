import numpy as np
from copy import deepcopy

def cosine_r6(m1,m2):
    corr = m1.dot(m2)/np.sqrt(m1.dot(m1)*m2.dot(m2))
    return corr

def minArc(x1,x2):
    delta_angle = (x1-x2)%360.
    delta_angle = np.where(delta_angle>180,360-delta_angle,delta_angle)
    return delta_angle.sum()

def kagan_angle(m1,m2):
    tensor1 = full_tensor(*m1)
    tensor2 = full_tensor(*m2)
    kagan = calc_theta(tensor1, tensor2)
    return kagan

def full_tensor(rr,tt,pp,rt,rp,tp):
    tensor = np.array([[rr,rt,rp],[rt,tt,tp],[rp,tp,pp]])
    return tensor

def calc_theta(vm1, vm2):
    """Calculate angle between two moment tensor matrices.
    Args:
        vm1 (ndarray): Moment Tensor matrix (see plane_to_tensor).
        vm2 (ndarray): Moment Tensor matrix (see plane_to_tensor).
    Returns:
        float: Kagan angle (degrees) between input moment tensors.
    """
    # calculate the eigenvectors of either moment tensor
    V1 = calc_eigenvec(vm1)
    V2 = calc_eigenvec(vm2)

    # find angle between rakes
    th = ang_from_R1R2(V1, V2)

    # calculate kagan angle and return
    for j in range(3):
        k = (j + 1) % 3
        V3 = deepcopy(V2)
        V3[:, j] = -V3[:, j]
        V3[:, k] = -V3[:, k]
        x = ang_from_R1R2(V1, V3)
        if x < th:
            th = x
    return th * 180. / np.pi

def calc_eigenvec(TM):
    """  Calculate eigenvector of moment tensor matrix.
    Args:
        ndarray: moment tensor matrix (see plane_to_tensor)
    Returns:
        ndarray: eigenvector representation of input moment tensor.
    """

    # calculate eigenvector
    V, S = np.linalg.eigh(TM)
    inds = np.argsort(V)
    S = S[:, inds]
    S[:, 2] = np.cross(S[:, 0], S[:, 1])
    return S


def ang_from_R1R2(R1, R2):
    """Calculate angle between two eigenvectors.
    Args:
        R1 (ndarray): eigenvector of first moment tensor
        R2 (ndarray): eigenvector of second moment tensor
    Returns:
        float: angle between eigenvectors
    """
    return np.arccos(np.clip((np.trace(np.dot(R1, R2.transpose())) - 1.) / 2.,-1,1))

