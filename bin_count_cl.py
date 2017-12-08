# -*- coding: utf-8 -*-
"""
Created on Thu Oct 2 01:49:21 2017

@author: Pasztor Balazs
"""
import numpy as np
import pyopencl as cl
import math
import time
import visvis as vv
from operator import itemgetter

from itertools import cycle
from vispy import app as vispyapp
from vispy import scene
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform


class mytimer:
    def __init__(self):
        self.last_time = time.perf_counter()
        #self.last_time = time.clock()
        self.times = [['process', 'timed', 'started', 'ended']]

    def measure(self, str):
        cur_time = time.perf_counter()
        #cur_time = time.clock()
        elapsed_time = cur_time - self.last_time
        self.times += [[str, elapsed_time, self.last_time, cur_time]]
        self.last_time = cur_time

    def print_times(self):
        print('{0[0]:40}: {0[1]:7.7}, {0[2]:7.7}, {0[3]:7.7}'.format(self.times[0]))
        for t in self.times[1:]:
            print('{0:40}: {1:7.4f}, {2:7.4f}, {3:7.4f}'.format(t[0], t[1], t[2], t[3]))


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def convert_np_data(rawdata):
    datadim = len(rawdata)
    datalength = len(rawdata[0])
    data = np.zeros((datadim, datalength))
    for d in range(datadim):
        data[d] = rawdata[d]
    return data


def bincount_CPU(data, bincounts = [10, 10]):
    data = convert_np_data(data)
    datadim = len(data)
    datalength = len(data[0])
    #data = np.zeros((datadim, datalength))
    #for d in range(datadim):
    #    data[d] = rawdata[d]

    #Min/Max of data per dimensions
    dataMins = np.zeros(datadim)
    dataMaxs = np.zeros(datadim)

    for d in range(datadim):
        dataMins[d] = np.amin(data[d])
        dataMaxs[d] = np.amax(data[d])

    # BINNING start
    bindim = len(bincounts)

    if(datadim < bindim):
        print('The specified bins have more dimensions than the data.',
              ' Extra dimenions will be ignored during binning')
        bincounts = bincounts[:datadim]
        bindim = datadim

    #for d in range (bindim):
    #    #if less bincounts are specified than the dimension of the data
    #    if len(binCounts) < (d + 1):
    #        binCounts.append(1)

    #calculating the widths of the bins
    binwidths = np.zeros(datadim)
    for d in range (bindim):
        binwidths[d] = (dataMaxs[d]-dataMins[d])/(bincounts[d]-2)

    #creating the bin-summarize array, with increased size for guards, NaNs, missing
    bins = np.zeros(tuple(v+0 for v in bincounts))

    for e in range(datalength):
        currbinnum = list()
        for d in range(bindim):
            currbinnum += [int((data[d, e]-dataMins[d])/binwidths[d]) + 1 ]
        # SUMMERIZE -- currently count
        bins[tuple(currbinnum)] += 1
    return bins


def show_visvis(bins_h, axislabels=None):
    app = vv.use()

    # Load volume
    vol = bins_h

    vv.figure(1); vv.clf()
    RS = ['ray']
    tt = []
    a = vv.subplot(1,1,1)
    t = vv.volshow(vol)
    t.interpolate = False
    #vv.title('Volume renderer')
    t.colormap = vv.CM_HOT
    t.renderStyle = 'ray'
    tt.append(t)

    if axislabels is not None:
        a.axis.xLabel = axislabels[0].replace('_', ' ')
        a.axis.yLabel = axislabels[1].replace('_', ' ')
        a.axis.zLabel = axislabels[2].replace('_', ' ')

    # Create colormap editor in first axes
    cme = vv.ColormapEditor(vv.gcf(), *tt[:])

    # Run app
    app.Create()
    app.Run()

# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


class TransRare(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        if (t >= 0.0001){        
            float c = 0.3 + 0.7 * t;
            return vec4(c, c, c, 0.1);
        }
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    """


class TransFireRare(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        if (t >= 0.0001){
            return vec4(pow(t, 0.5), t, t*t, 0.1);
        }
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    """


class TransFireRareOld(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        if (t >= 0.0001){
            return vec4(0.3 + 0.7* pow(t, 0.5), 0.3 + 0.7* t, 0.3 + 0.7* t*t, 0.1);
        }
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    """

# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays(), TransRare(), TransFireRare()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


def show_vispy(bins_h):
    vol = bins_h

    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', app='pyqt5', size=(800, 600), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    emulate_texture = False     #OpenGL ES miatt kell a true csak

    # Create the volume visuals, only one is visible
    volume1 = scene.visuals.Volume(vol, parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.
    cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
    cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name='Turntable')
    cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')
    view.camera = cam1  # Select turntable at first

    # Create an XYZAxis visual
    axis = scene.visuals.XYZAxis(parent=view)
    s = STTransform(translate=(250, 250), scale=(550, 550, 550, 1))
    affine = s.as_matrix()
    axis.transform = affine

    # Implement axis connection with cam2
    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        if event.button == 1 and event.is_dragging:
            axis.transform.reset()

            axis.transform.rotate(cam2.roll, (0, 0, 1))
            axis.transform.rotate(cam2.elevation, (1, 0, 0))
            axis.transform.rotate(cam2.azimuth, (0, 1, 0))

            axis.transform.scale((50, 50, 0.001))
            axis.transform.translate((50., 50.))
            axis.update()


    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global opaque_cmap, translucent_cmap
        if event.text == '1':
            cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
            view.camera = cam_toggle.get(view.camera, cam2)
            print(view.camera.name + ' camera')
            if view.camera is cam2:
                axis.visible = True
            else:
                axis.visible = False
        elif event.text == '2':
            print('2-pressed')
            methods = ['mip', 'translucent', 'iso', 'additive']
            method = methods[(methods.index(volume1.method) + 1) % 4]
            print("Volume render method: %s" % method)
            cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
            volume1.method = method
            volume1.cmap = cmap
        elif event.text == '3':
            if volume1.method in ['mip', 'iso']:
                cmap = opaque_cmap = next(opaque_cmaps)
            else:
                cmap = translucent_cmap = next(translucent_cmaps)
            volume1.cmap = cmap
        elif event.text == '0':
            cam1.set_range()
            cam3.set_range()
        elif event.text != '' and event.text in '-+':
            s = -0.025 if event.text == '+' else 0.025
            volume1.threshold += s
            th = volume1.threshold
            print("Isosurface threshold: %0.3f" % th)

    vispyapp.run()


mt = mytimer()

kernel_text_skeleton = """
inline int get_bin_number(
      const float value
    , const float binwidth
    , const float bin_origin
    , const int bin_max)
{
    if(value != value){
        return -1;
    }
    int bin_num = (int)((value - bin_origin) / binwidth);
    if(bin_num < 0 || bin_num >= bin_max){
        return -1;
    }
    return bin_num;
}

inline size_t get_bin_number_incl_nan(
      const float value
    , const float binwidth
    , const float bin_origin
    , const int bin_max)
{
    if(value != value){
        return 0;
    }
    int bin_num = (int)((value - bin_origin) / binwidth) + 1;
    if(bin_num < 0 || bin_num >= bin_max){
        return 0;
    }
    return bin_num;
}

__kernel void globalcount_nd(
^^GLOBALCOUNT_ND_KERNEL_ARGS
    const int data_length ,
    __global volatile int *bins)
{
    size_t g_id = get_global_id(0);

    for(size_t di = g_id; di < data_length; di += ^^CHUNK_LENGTH){
^^GLOBALCOUNT_ND_KERNEL_WORK
    }
}
"""


kernels_text = """
__kernel void count_on_sorted(
    global const float* data,
    global const int* indexes,
    const int bins_length ,
    global int *count)
{
    size_t g_id = get_global_id(0);
    if(g_id > bins_length -1){
        return;
    }

    size_t idx_from = indexes[g_id];
    size_t idx_to = indexes[g_id + 1];

    count[g_id + 1] = idx_to - idx_from;
}


__kernel void avg_on_sorted(
    global const float* data,
    global const int* indexes,
    const int bins_length ,
    global float *avg_b,
    const float when_zero)
{
    size_t g_id = get_global_id(0);
    if(g_id > bins_length -1){
        return;
    }

    size_t idx_from = indexes[g_id];
    size_t idx_to = indexes[g_id + 1];
    int count = idx_to - idx_from;
    
    if(count <= 0){
        avg_b[g_id] = when_zero;
        return;
    }
    float mult = 1/(float)(idx_to - idx_from);
    float avg = 0;
    for(int i = idx_from; i < idx_to; ++i){
        avg += data[i] * mult;
    }
    avg_b[g_id] = avg;
}

__kernel void percentile_on_sorted(
    global const float* data,
    global const int* indexes,
    const int bins_length ,
    global float *percentile_b,
    const float percentile,
    const float when_zero)
{
    size_t g_id = get_global_id(0);
    if(g_id > bins_length -1){
        return;
    }

    size_t idx_from = indexes[g_id];
    size_t idx_to = indexes[g_id + 1];
    int count = idx_to - idx_from;
    
    if(count <= 0){
        percentile_b[g_id] = when_zero;
        return;
    }

    int i = (int)((float)(count) * percentile);
    percentile_b[g_id] = data[idx_from + i];
}

"""


def bincount_cl(data_h, spec_bin_limits=[[None, None]], bin_counts = [50, 50]):
    global mt
    mt.measure('bincount_cl called')

    datadim = len(data_h)
    datalength = len(data_h[0])
    dataMins = np.zeros(datadim)
    dataMaxs = np.zeros(datadim)
    #print(datalength)

    bindim = len(bin_counts)
    if(datadim < bindim):
        print('The specified bins have more dimensions than the data.',
              ' Extra dimenions will be ignored during binning')
        bin_counts = bin_counts[:datadim]
        bindim = datadim

    for i in range(bindim):
        if len(data_h[i]) != datalength:
            print('Length of data columns are not equal!')
            return

    #if data mins / maxs are not specified:
    for d in range(bindim):
        if len(spec_bin_limits) > d and spec_bin_limits[d][0] is not None:
            dataMins[d] = spec_bin_limits[d][0]
        else:
            dataMins[d] = np.amin(data_h[d])

        if len(spec_bin_limits) >= d and spec_bin_limits[d][1] is not None:
            dataMaxs[d] = spec_bin_limits[d][1]
        else:
            dataMaxs[d] = np.amax(data_h[d])

    mt.measure('Looking for Data minimums and maximums.')

    data_d = [None] * datadim
    for i in range(datadim):
        if data_h[i].dtype == np.int32 or data_h[i].dtype == np.float32:
            data_d[i] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_h[i].astype(dtype=np.float32))
        else:
            print("Unsopported type at dimension "+str(i)+" (starting from 0).")
            return
    mt.measure('Uploading data to the GPU.')

    binwidths = np.zeros(datadim)
    for d in range(bindim):
        binwidths[d] = (dataMaxs[d] - dataMins[d])/(bin_counts[d]-2)

    bins_buffer_length = 1
    for i in range(len(bin_counts)):
        bins_buffer_length *= bin_counts[i]

    bins_h = np.zeros(bins_buffer_length, dtype=np.int32)
    bins_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bins_h)
    mt.measure('Creaing the buffer for the bins')

    #kernels_text = open('./kernels/kernels.cl').read()
    global kernel_text_skeleton
    kernels_text = kernel_text_skeleton
    kernel_seq_part_length = 128
    kernel_parallel_amount = math.ceil(datalength/kernel_seq_part_length)

    kernelargs_str = ''
    for d in range(datadim):
        type = ''.join([char for char in str(data_h[d].dtype) if not char.isdigit()])
        kernelargs_str += '    __global const '+type +' *axis' + str(d) + ', \n'

    kernels_text = kernels_text.replace('^^GLOBALCOUNT_ND_KERNEL_ARGS', kernelargs_str)

    kernelwork_str = ''
    kernelwork_indexing = ''
    for d in range(bindim):
        kernelwork_str += '        int bin' + str(d) + ' = get_bin_number(axis' + str(d) + '[di], '
        kernelwork_str += str(binwidths[d]) + ', ' + str(dataMins[d]) + ', ' + str(bin_counts[d] - 1) + ');\n'
        if d >= 1:
            kernelwork_indexing += ' + '
        kernelwork_indexing += 'bin' + str(d)
        if d >= 1:
            axismult = 1
            for i in range(d):
                axismult *= bin_counts[i]
            kernelwork_indexing += ' * ' + str(axismult)

    kernelwork_str += '\n        if('
    for d in range(bindim):
        if d >= 1:
            kernelwork_str += '&& '
        kernelwork_str += 'bin' + str(d) + ' > 0 '
    kernelwork_str += ') {'
    kernelwork_str += '\n            atomic_inc(&bins[' + kernelwork_indexing + ']);'
    kernelwork_str += '\n        }'
    kernels_text = kernels_text.replace('^^GLOBALCOUNT_ND_KERNEL_WORK', kernelwork_str)

    kernels_text = kernels_text.replace('^^CHUNK_LENGTH', str(kernel_parallel_amount))
    kernels_text = kernels_text.replace('^^DATA_LENGTH', str(datalength))

    #print(kernels_text)
    #with open("built_kernel_text.txt", "w") as text_file:
    #    text_file.write(kernels_text)

    prg = cl.Program(ctx, kernels_text).build()
    params = [ queue
            , tuple([math.ceil(datalength/kernel_seq_part_length)])
            , None]
    params += data_d
    params += [np.int32(datalength)]
    params += [bins_d]
    mt.measure('Reading in, and assembling the kernel')

    prg.globalcount_nd(*params).wait()
    mt.measure('Binning (count)')

    cl.enqueue_copy(queue, bins_h, bins_d)
    bins_h = bins_h.reshape(tuple(np.flip(bin_counts, 0)))

    mt.measure('Reading back the result')
    return bins_h


def bin_cpu(data_h, spec_bin_limits=[[None, None]], bin_counts = [50, 50]):
    #global mt
    #mt.measure('bincount_cl called')

    datadim = len(data_h)
    datalength = len(data_h[0])
    dataMins = np.zeros(datadim)
    dataMaxs = np.zeros(datadim)
    #print(datalength)

    bindim = len(bin_counts)
    if(datadim < bindim):
        print('The specified bins have more dimensions than the data.',
              ' Extra dimenions will be ignored during binning')
        bin_counts = bin_counts[:datadim]
        bindim = datadim

    for i in range(bindim):
        if len(data_h[i]) != datalength:
            print('Length of data columns are not equal!')
            return

    #if data mins / maxs are not specified:
    for d in range(bindim):
        if len(spec_bin_limits) > d:
            if spec_bin_limits[d][0] is not None:
                dataMins[d] = spec_bin_limits[d][0]
            else:
                dataMins[d] = np.amin(data_h[d])
        else:
            dataMins[d] = np.amin(data_h[d])

        if len(spec_bin_limits) >= d:
            if spec_bin_limits[d][1] is not None:
                dataMaxs[d] = spec_bin_limits[d][1]
            else:
                dataMaxs[d] = np.amax(data_h[d])
        else:
            dataMaxs[d] = np.amax(data_h[d])

    binwidths = np.zeros(datadim)
    for d in range(bindim):
        binwidths[d] = (dataMaxs[d] - dataMins[d])/(bin_counts[d]-2)

        bins_buffer_length = 1
    for i in range(len(bin_counts)):
        bins_buffer_length *= bin_counts[i]

    bin_borders = np.zeros(bins_buffer_length, dtype=np.int32).reshape(tuple(bin_counts))

    data_tup = [0]*datalength
    for i in range(datalength):
        tmplist = [0]*datadim
        for d in range(datadim):
            tmplist[d] = data_h[d][i]
        data_tup[i] = tuple(tmplist)

    sort_and_search(data_tup, 0, datalength, bin_counts, bin_borders, dataMins, dataMaxs, 0, [])

    for i in range(datalength):
        for d in range(datadim):
            data_h[d][i] = data_tup[i][d]

    return tuple([data_h, bin_borders])


def sort_and_search(data_tup, from_idx, to_idx, bin_counts, bin_bounds, dataMins, dataMaxs, d, bin_idx_so_far):
    if d < len(bin_counts):
        sort_reverse = False
        if dataMins[d] > dataMaxs[d]:
            sort_reverse = True

        data_tup[from_idx : to_idx] = sorted(data_tup[from_idx : to_idx], key=itemgetter(d), reverse=sort_reverse)

        i = from_idx
        j = 0
        next_bin_bound_val = dataMins[d]
        while i < to_idx and j < (bin_counts[d] - 1):
            if not sort_reverse:
                if data_tup[i][d] > next_bin_bound_val:
                    j += 1
                    bin_idx_list = bin_idx_so_far + [j]
                    bin_idx_list += [0] * (len(bin_counts) - len(bin_idx_list))
                    bin_bounds[tuple(bin_idx_list)] = i
                    next_bin_bound_val = dataMins[d] + (dataMaxs[d] - dataMins[d]) * j / (bin_counts[d]-2)          #j = [0 .. (counts-1)] miatt kell a bin_counts.. -> -1 <- izé
                else:
                    i += 1
            else:
                if data_tup[i][d] < next_bin_bound_val:
                    j += 1
                    bin_idx_list = bin_idx_so_far + [j]
                    bin_idx_list += [0] * (len(bin_counts) - len(bin_idx_list))
                    bin_bounds[tuple(bin_idx_list)] = i
                    next_bin_bound_val = dataMins[d] + (dataMaxs[d] - dataMins[d]) * j / (bin_counts[d]-2)          #j = [0 .. (counts-1)] miatt kell a bin_counts.. -> -1 <- izé
                else:
                    i += 1

        while j < bin_counts[d] - 1:
            j += 1
            bin_idx_list = bin_idx_so_far + [j]
            bin_idx_list += [0] * (len(bin_counts) - len(bin_idx_list))
            bin_bounds[tuple(bin_idx_list)] = i

        if d+1 < len(bin_counts) or d+1 < len(data_tup[0]):
            for i in range(1, bin_counts[d]-1):
                idx_list_from = bin_idx_so_far + [i]
                idx_list_from += [0] * (len(bin_counts) - len(idx_list_from))
                new_from_idx = bin_bounds[tuple(idx_list_from)]

                idx_list_to = bin_idx_so_far + [i + 1]
                idx_list_to += [0] * (len(bin_counts) - len(idx_list_to))
                new_to_idx = bin_bounds[tuple(idx_list_to)]

                bin_idx_list = bin_idx_so_far + [i]
                sort_and_search(data_tup, new_from_idx, new_to_idx, bin_counts, bin_bounds, dataMins, dataMaxs, d+1, bin_idx_list)
    else:
        if d < len(data_tup[0]):
            data_tup[from_idx : to_idx] = sorted(data_tup[from_idx : to_idx], key=itemgetter(d))


def only_count_cl(data_bin_bord_tup):
    data_h = data_bin_bord_tup[0]
    orig_shape = data_bin_bord_tup[1].shape
    bin_borders_h = data_bin_bord_tup[1].reshape(tuple([data_bin_bord_tup[1].size, ]))

    datadim = len(data_h)
    datalength = len(data_h[0])

    data_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_h[-1].astype(dtype=np.float32))
    bin_borders_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bin_borders_h.astype(dtype=np.int32))

    result_h = np.zeros(bin_borders_h.size, dtype=np.int32)
    result_d = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=result_h)

    #kernels_text = open('./kernels/kernels2.cl').read()
    global kernels_text
    prg = cl.Program(ctx, kernels_text).build()
    params = [ queue
            , tuple([bin_borders_d.size-1])
            , None]
    params += [data_d]
    params += [bin_borders_d]
    params += [np.int32(bin_borders_h.size-1)]
    params += [result_d]

    prg.count_on_sorted(*params).wait()

    cl.enqueue_copy(queue, result_h, result_d)

    if len(orig_shape) == 1:
        return result_h.reshape(orig_shape)[1:-1].transpose()
    if len(orig_shape) == 2:
        return result_h.reshape(orig_shape)[1:-1, 1:-1].transpose()
    if len(orig_shape) == 3:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 4:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 5:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 6:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()


def only_avg_cl(data_bin_bord_tup, when_empty=0.0):
    data_h = data_bin_bord_tup[0]
    orig_shape = data_bin_bord_tup[1].shape
    bin_borders_h = data_bin_bord_tup[1].reshape(tuple([data_bin_bord_tup[1].size, ]))

    datadim = len(data_h)
    datalength = len(data_h[0])

    data_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_h[-1].astype(dtype=np.float32))
    bin_borders_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bin_borders_h.astype(dtype=np.int32))

    result_h = np.zeros(bin_borders_h.size, dtype=np.float32)
    result_d = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=result_h)

    #kernels_text = open('./kernels/kernels2.cl').read()
    global kernels_text
    prg = cl.Program(ctx, kernels_text).build()
    params = [ queue
            , tuple([bin_borders_d.size-1])
            , None]
    params += [data_d]
    params += [bin_borders_d]
    params += [np.int32(bin_borders_h.size-1)]
    params += [result_d]
    params += [np.float32(when_empty)]

    prg.avg_on_sorted(*params).wait()

    cl.enqueue_copy(queue, result_h, result_d)

    if len(orig_shape) == 1:
        return result_h.reshape(orig_shape)[1:-1].transpose()
    if len(orig_shape) == 2:
        return result_h.reshape(orig_shape)[1:-1, 1:-1].transpose()
    if len(orig_shape) == 3:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 4:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 5:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 6:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()


def only_percentile_cl(data_bin_bord_tup, percentile=50, when_empty=0.0):
    data_h = data_bin_bord_tup[0]
    orig_shape = data_bin_bord_tup[1].shape
    bin_borders_h = data_bin_bord_tup[1].reshape(tuple([data_bin_bord_tup[1].size, ]))

    if percentile < 0 or percentile > 100:
        print('Incorrect value for percentile! Only numbers from 0 to 100 are allowed!')
        return

    datadim = len(data_h)
    datalength = len(data_h[0])

    data_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_h[-1].astype(dtype=np.float32))
    bin_borders_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bin_borders_h.astype(dtype=np.int32))

    result_h = np.zeros(bin_borders_h.size, dtype=np.float32)
    result_d = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=result_h)

    #kernels_text = open('./kernels/kernels2.cl').read()
    global kernels_text
    prg = cl.Program(ctx, kernels_text).build()
    params = [ queue
            , tuple([bin_borders_d.size-1])
            , None]
    params += [data_d]
    params += [bin_borders_d]
    params += [np.int32(bin_borders_h.size-1)]
    params += [result_d]
    params += [np.float32(percentile / 100.0)]
    params += [np.float32(when_empty)]

    prg.percentile_on_sorted(*params).wait()

    cl.enqueue_copy(queue, result_h, result_d)

    if len(orig_shape) == 1:
        return result_h.reshape(orig_shape)[1:-1].transpose()
    if len(orig_shape) == 2:
        return result_h.reshape(orig_shape)[1:-1, 1:-1].transpose()
    if len(orig_shape) == 3:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 4:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 5:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()
    if len(orig_shape) == 6:
        return result_h.reshape(orig_shape)[1:-1, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1].transpose()


def only_min_cl(data_bin_bord_tup, when_empty=0.0):
    return only_percentile_cl(data_bin_bord_tup, 0, when_empty)


def only_max_cl(data_bin_bord_tup, when_empty=0.0):
    return only_percentile_cl(data_bin_bord_tup, 100, when_empty)
