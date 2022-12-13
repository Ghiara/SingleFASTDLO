import itertools
import numpy as np
import arrow
import cv2

from fastdlo.siam_net.nn_predict import NN
from fastdlo.seg_net.predict import SegNet
#########################################################################
####### import color Filter for color masking, modified by Y.Meng #######
#########################################################################
from fastdlo.colorFilter import ColorFilter

from fastdlo.siam_net.nn_dataset import AriadnePredictData
from fastdlo.proc.labelling import LabelsPred
import fastdlo.proc.utils as utils




class Pipeline():


    def __init__(self, checkpoint_siam, checkpoint_seg=None, img_w = 640, img_h = 480, colorRange = None, is_interpolation=False):

        self.network = NN(device="gpu", checkpoint_path=checkpoint_siam)
        ##############################################
        ############ Add color Filter instance #######
        ##############################################
        self.colorFilter = ColorFilter()
        if colorRange is not None:
            self.colorRange = colorRange
            self.colorFilter.colorRange = self.colorRange
        else:
            self.colorRange = None
            # self.colorFilter.colorRange = [((156,43,46),(180,255,255))] # by default detect red color items
        # interpolation request? by default False
        self.activate_interpolation = is_interpolation
        
        if checkpoint_seg is not None:
            self.network_seg = SegNet(model_name="deeplabv3plus_resnet101", checkpoint_path=checkpoint_seg, img_w=img_w, img_h=img_h)
        else:
            self.network_seg = None


    def run(self, source_img, mask_th = 127):
        '''
        used to compute the FASTDLO algorithm
        output:
        img_out     masked image with mask green other place black
        splines     splines dict{'points':list,'der':list, 'der2':list, 'radius':list}
        path_final  the DLO detection point set dict {'points':list, ..}
        '''
        t0 = arrow.utcnow()

        # get image MASK

        if self.colorRange is not None:
            # for case of color filtering required
            color_filtered_img = self.colorFilter(source_img)
            mask_img = self.network_seg.predict_img(color_filtered_img)
        else:
            # No color filtering required, use origin image
            mask_img = self.network_seg.predict_img(source_img)

        # the background segmented mask value greater than threshold, will be labbled as white(255), otherwise black(0)
        mask_img[mask_img > mask_th] = 255
        mask_img[mask_img != 255] = 0
        #mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8))

        seg_time = (arrow.utcnow() - t0).total_seconds() * 1000

        img_out, dlo_mask_pointSet, dlo_path,times = self.process(source_img=source_img, mask_img=mask_img, mask_th=mask_th)

        times["seg_time"] = seg_time

        return img_out, dlo_mask_pointSet, dlo_path



    def process(self, source_img, mask_img, mask_th=127):
        '''
        binary mask_img, with white(255), black(0)
        '''
        t0 = arrow.utcnow() #####################

        # PRE-PROCESSING
        lp = LabelsPred()
        # labelling
        rv = lp.compute(source_img=source_img, mask_img=mask_img, timings=False, mask_threshold=mask_th)
        nodes, single_nodes, pred_edges, vertices_dict, radius_dict, ints_dict  = \
            rv["nodes"], rv["single_nodes"], rv["pred_edges"], rv["vertices"], rv["radius"], rv["intersections"]
        
        data_dict = {"nodes": nodes, "pred_edges_index": pred_edges}

        pre_time = (arrow.utcnow() - t0).total_seconds() * 1000
        #cprint("time pre-processing: {0:.4f} ms".format((arrow.utcnow() - t0).total_seconds() * 1000), "yellow")


        # PATHS PREDICTION FOR PRED_EDGES AND NN
        paths_final, vertices_dict_excluded, preds_sorted, pred_time = self.predictAndMerge(data_dict, vertices_dict, radii_dict=radius_dict)


        # TRYING TO MERGE EXCLUDED SEGMENTS
        int_excluded_dict = self.processExcludedPaths(vertices_dict_excluded, intersections_init=ints_dict, vertices_dict=vertices_dict)
        if int_excluded_dict:
            #print("EXCL: ", int_excluded_dict)


            #print("PREDCTING EXCLUDED SEGMENTS...")
            pred_edges_2 = lp.computeExcluded(int_excluded_dict) #computeLocalPredictionEdgesExcluded(int_excluded_dict)

            #print("pred_edges_2: ", pred_edges_2)
            data_dict2 = {"nodes": nodes, "pred_edges_index": pred_edges_2}
 
            paths_final_2, vertices_dict_excluded_2, preds_sorted_2, pred_time_2 = self.predictAndMerge(data_dict2, vertices_dict_excluded, radii_dict=radius_dict)

            # merge the two dicts
            paths_last_key = len(paths_final.keys())
            for it, v in enumerate(paths_final_2.values()):
                paths_final[paths_last_key + it] = v

            # considering excluded paths
            paths_last_key = len(paths_final.keys())
            for k, v in vertices_dict.items():
                if k in vertices_dict_excluded.keys() and k in vertices_dict_excluded_2.keys():
                    paths_final[paths_last_key + k] = {"points": v["pos"], "radius": radius_dict[k]}  


            preds_all_out = [(p["node_0"], p["node_1"], p["score"]) for p in preds_sorted]
            preds_all_out.extend([(p["node_0"], p["node_1"], p["score"]) for p in preds_sorted_2])

        else:
            # considering excluded paths
            paths_last_key = len(paths_final.keys())
            counter = 0
            for k, v in vertices_dict.items():
                if k in vertices_dict_excluded.keys():
                    paths_final[paths_last_key + counter] = {"points": v["pos"], "radius": radius_dict[k]}   
                    counter += 1

            preds_all_out = [(p["node_0"], p["node_1"], p["score"]) for p in preds_sorted]


        # SPLINES FOR MODELLING DLOS
        splines = utils.computeSplines(paths_final, key="points")
        '''
        Notation by Y.Meng
        splines a nested dict with keys indicate the series number of identified splines, 
        under each spline, 'points' saved the spline point positions in form of (x,y)list
        'der', 
        'der2',
        'radius'
        '''
        ##########################################################################################

        #ts = arrow.utcnow()
        try:
            int_splines = utils.intersectionSplines(splines, paths_final, single_nodes, img=source_img)
        except:
            int_splines = None
        #cprint("time intersections splines: {0:.4f} ms".format((arrow.utcnow() - ts).total_seconds() * 1000), "yellow")
        
        # use the clustered splines point sets to create colored mask
        '''
        Notation by Y.Meng
        function colorMasks has been modified, use green identify all the mask objects, detect DLO as single
        '''
        colored_mask = utils.colorMasks(splines, shape=mask_img.shape, mask_input=None)

        if int_splines:
            # update with the score, if mask provided draw already the final mask

            #ti = arrow.utcnow()
            int_splines = utils.intersectionScoresFromColor(int_splines, nodes, image=source_img, colored_mask=colored_mask)
            #int_splines = utils.intersectionScoresFromColor(int_splines, nodes, image=source_img, colored_mask=None)
            #cprint("time intersections scores with color mask: {0:.4f} ms".format((arrow.utcnow() - ti).total_seconds() * 1000), "yellow")

        ##########################################################################################
        

        tot_time = (arrow.utcnow() - t0).total_seconds()*1000
        times = {"tot_time": tot_time, "proc_time": pre_time, "skel_time": rv["time"]["skel"], "pred_time": pred_time}
        #cprint("***** ALL PROCESSING TIME: {0:.4f} ms *****".format(tot_time), "yellow")

        # undetected mask labelled as black
        colored_mask[mask_img < mask_th] = (0, 0, 0)
        '''
        modified by Y.Meng
        reunion the color mask and path
        '''
        dlo_mask_pointSet = self.merge_dict(spline_dict=splines)
        dlo_path = self.pathConnection(paths_final)

        # add linear interpolation function
        if self.activate_interpolation:
            interpolated_dlo_path = self.pathInterpolation(dlo_path)
            return colored_mask, dlo_mask_pointSet, interpolated_dlo_path, times
        else:
            return colored_mask, dlo_mask_pointSet, dlo_path, times
    

    def merge_dict(self, spline_dict = None):
        '''
        Developed by Y.Meng
        merge the nested spline dict or final path dict
        '''

        # merged the splines nested dict into one dict instance
        if spline_dict != None:
            points_spline = []

            for idx, dict in enumerate(spline_dict.items()):
                points_spline.extend(dict[1]['points'])


        return points_spline

    def pathInterpolation(self, data):
            '''
            developed by Y.Meng, :
            use linear interpolation to fill the vacancy of DLO path 
            '''
            for i in range(len(data)):
                pos = data[i]
                if i != len(data)-1:
                    pos2 = data[i+1]
                    # detect whether the points in between far away, if not, then connect them with line segment
                    length = np.sqrt(((pos[0]-pos2[0])**2)+((pos[1]-pos2[1])**2))
                    # Notation the closest point distance is about 5, so the interpolated points vancancy should be at least 10
                    if length >= 10:
                        # calculate the number of points should be interpolated
                        num_interpolation = int(length / 5 - 1)
                        assert num_interpolation > 0
                        
                        x = [pos[0], pos2[0]]
                        y = [pos[1], pos2[1]]
                        # find linear fitting curve
                        a, b = np.polyfit(x, y, deg=1)
                        
                        for idx in range(num_interpolation):
                            # calculate interpolation points
                            x_interpolate = pos[0] + (pos2[0] - pos[0])*((num_interpolation - idx)/(num_interpolation+1))
                            y_interpolate = a * x_interpolate + b
                            data.insert(i+1,np.array([int(x_interpolate), int(y_interpolate)]))
                        
                    else:
                        pass
                else:
                    pos2 = None
                return data

            
    def pathConnection(self, paths_dict):
        '''
        developed by Y.Meng, used to connect the path line lists
        basic idea is K-Nearest Neighboor where K == 1
        
        '''
        points_path = []

        for i, item in enumerate(paths_dict.values()):
            points_path.append(item['points'])
        
        spline_num = len(points_path)

        if spline_num == 1:
            return points_path[0]
        elif spline_num == 0:
            points_path.append(np.array([0,0]))
            return points_path
        else:
            count = spline_num - 1

            # take the first spline
            final_path = points_path[0]

            while count > 0:
                head = final_path[0]
                tail = final_path[-1]
                map1 = []
                map2 = []
                # calculate the closest distance btw head & tail to the first spline, and find smallest one
                for path in points_path[1:]:
                    son_head = path[0]
                    son_tail = path[-1]
                    length1 = np.sqrt(((head[0]-son_tail[0])**2)+((head[1]-son_tail[1])**2))
                    length2 = np.sqrt(((son_head[0]-tail[0])**2)+((son_head[1]-tail[1])**2))
                    map1.append(length1)
                    map2.append(length2)
                
                map = np.array([map1,map2])
                h,w = map.shape
                pos = map.argmin()
                idx_row, idx_col = pos // w, pos % w
                # find closest distance and its connection position
                if idx_row == 0:
                    points_path[idx_col+1].extend(points_path[0])
                    final_path = points_path[idx_col+1]
                    points_path[0] = final_path
                    del points_path[idx_col+1]
                else:
                    final_path.extend(points_path[idx_col+1])
                    del points_path[idx_col+1]
                points_path[0] = final_path

                count -= 1
        return final_path


    def solveIntersections(self, preds_sorted, nodes, vertices_dict, debug=False):
        '''
        used the most possible probability to connect the endpoints of intersection areas
        '''
        nodes_done = []
        segments_completed = []
        for it in range(len(preds_sorted)):
            #print("IT: ", it)

            p = preds_sorted[it]
            node_0 = p["node_0"]
            node_1 = p["node_1"]

            if node_0 in nodes_done or node_1 in nodes_done:
                #print(f"skipping index {it}, one or more nodes already processed!")
                continue

            node_0_seg_id = nodes[node_0]["segment"]
            node_1_seg_id = nodes[node_1]["segment"]
            #print(node_0_seg_id, node_1_seg_id)


            nodes_list_0, nodes_list_1 = None, None
            seg_founds_it, seg_founds_ids = [], []
            for it, seg_compl in enumerate(segments_completed):
                if node_0_seg_id in seg_compl["ids"]:
                    nodes_list_0 = seg_compl["nodes"]
                    seg_founds_it.append(it)
                    seg_founds_ids.append(seg_compl["ids"])

                elif node_1_seg_id in seg_compl["ids"]:
                    nodes_list_1 = seg_compl["nodes"]
                    seg_founds_it.append(it)
                    seg_founds_ids.append(seg_compl["ids"])

            #print("seg_founds_it ", seg_founds_it)
            #print("seg_founds_ids ", seg_founds_ids)

            if nodes_list_0 is None:
                nodes_list_0 = vertices_dict[node_0_seg_id]["nodes"]
            if nodes_list_1 is None:
                nodes_list_1 = vertices_dict[node_1_seg_id]["nodes"]

            index_0 = nodes_list_0.index(node_0)
            index_1 = nodes_list_1.index(node_1)

            merged = []
            if index_0 == 0:
                merged.extend(nodes_list_0[::-1])
            else:
                merged.extend(nodes_list_0)

            if index_1 != 0:
                merged.extend(nodes_list_1[::-1])
            else:
                merged.extend(nodes_list_1)

            # update seg ids
            all_ids = [node_0_seg_id, node_1_seg_id]
            if seg_founds_ids:
                for seg_ids in seg_founds_ids:
                    all_ids.extend(seg_ids)
                all_ids = list(set(all_ids))


            # delete old segments
            #print(seg_founds_it)
            for it in reversed(seg_founds_it):
                del segments_completed[it]

            # add new merged segment
            segments_completed.append({"ids": all_ids, "nodes": merged})

            nodes_done.append(node_0)
            nodes_done.append(node_1)

            #print(segments_completed)
        
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        #for seg in segments_completed:
        #    print(seg)
        return segments_completed, nodes_done


    def computeListPoints(self, data, nodes_dict, vertices_dict, radii_dict):
        points_dict = {}
        for it, value in enumerate(data):
            radius = [radii_dict[idx] for idx in value["ids"]]
            points = [nodes_dict[nid]["pos"] for nid in value["nodes"]]
                
            points_dict[it] = {"points": points, "radius": np.mean(radius), "nodes": value["nodes"]}

        excluded_vertices_dict = {}
        for k, v in vertices_dict.items():
            data_keys = list(set([v for value in data for v in value["ids"]]))
            if k not in data_keys:
                excluded_vertices_dict[k] = v
                #print("excluded segment: ", k)

        return points_dict, excluded_vertices_dict


    def processExcludedPaths(self, paths_excluded, intersections_init, vertices_dict, max_hops = 5, debug=False):
        paths_keys = list(paths_excluded.keys())
        if debug: print("paths keys: ", paths_keys)

        # associate excluded paths to intersections
        paths_with_int = {}
        for k, v in intersections_init.items():
            r = list(set(paths_keys).intersection(set(v["segments"])))
            if r:
                paths_with_int[k] = r

        if debug: print("paths_with_int: \n", paths_with_int)

        # check if a common segment exist
        combinations_list = list(map(list, itertools.combinations(list(paths_with_int.keys()), 2)))
        candidates_dict = {}
        for it, (c0,c1) in enumerate(combinations_list):
            seq0 = intersections_init[c0]["segments"]
            seq1 = intersections_init[c1]["segments"]
            r = list(set(seq0).intersection(set(seq1)))
            if len(r) == 1:
                if r[0] in vertices_dict:
                    if len(vertices_dict[r[0]]) < max_hops:
                        mean_int = np.mean([intersections_init[c0]["point"], intersections_init[c1]["point"]], axis=0)
                        segments_ids = [paths_with_int[c0][0], paths_with_int[c1][0]]
                        candidates_dict[it] = {"point": mean_int, "segments": segments_ids}
                    else:
                        print("max hops reached!")

        if debug: print("candidates_dict: ", candidates_dict)

        return candidates_dict


    def predictAndMerge(self, graph_dict, vertices_dict, radii_dict, debug=False):
        '''
        use the probability list to connect the intersection areas
        '''
        t0 = arrow.utcnow()

        data_network = AriadnePredictData.getAllPairs(graph_dict["nodes"], graph_dict["pred_edges_index"])

        #print(data_network) 

        t0 = arrow.utcnow()

        # predict edges at intersections
        preds = self.network.predictBatch(data_network, threshold=0.2)

        pred_time = (arrow.utcnow() - t0).total_seconds() * 1000

        # predictions sorted (from best to worse)
        preds_sorted = sorted(preds, key=lambda d: d['score'], reverse=True) 

        if debug:
            print("==================\nPREDICTIONS NN:")
            for p in preds_sorted: 
                print("None 0: {0}, Node 1: {1}, Score: {2:.5f}".format(p["node_0"], p["node_1"], p["score"]))
            print("==================")

        # solve intersections 
        data, nodes_done = self.solveIntersections(preds_sorted, graph_dict["nodes"], vertices_dict)
        
        # retrieve final paths and excluded segments
        paths_final, paths_excluded = self.computeListPoints(data, graph_dict["nodes"], vertices_dict, radii_dict)

        #cprint("time predict_merge: {0:.4f} ms".format((arrow.utcnow() - t0).total_seconds() * 1000), "yellow")
        return paths_final, paths_excluded, preds_sorted, pred_time



        