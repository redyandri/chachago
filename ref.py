from fastapi import FastAPI, File
from starlette.requests import Request
import pickle
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import uuid
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi import APIRouter
from fastapi.routing import APIRoute
from typing import Callable
from pydantic import BaseModel
from fastapi import Body
import uvicorn
import datetime
import io
import json
from typing import List
from typing import Dict
from pydantic import BaseModel
import queue
from docxtpl import DocxTemplate
from io import StringIO
from starlette.responses import StreamingResponse
# import redis
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
import logging
from datetime import datetime
import math
from neo4j import GraphDatabase
from nltk.tag import CRFTagger
from decouple import config
import colorsys

app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# app = FastAPI()

# # Handle CORS
# class CORSHandler(APIRoute):
#     def get_route_handler(self) -> Callable:
#         original_route_handler = super().get_route_handler()

#         async def preflight_handler(request: Request) -> Response:
#             if request.method == 'OPTIONS':
#                 response = Response()
#                 response.headers['Access-Control-Allow-Origin'] = '*'
#                 response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
#                 response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
#             else:
#                 response = await original_route_handler(request)

#         return preflight_handler

# router = APIRouter(route_class=CORSHandler)
# app.include_router(router)


tagger = CRFTagger()
tagger.set_model_file(r"model/all_indo_man_tag_corpus_model.crf.tagger")


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# default_stopwords = StopWordRemoverFactory().get_stop_words()
# additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
# dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
# id_stopword = StopWordRemover(dictionary)

en_stemmer = PorterStemmer()
en_stopwords = nltk.corpus.stopwords.words('english')

df_id_stopword = pd.read_csv("data/stopwordbahasa.csv", header=None)
id_stopword = df_id_stopword[0].to_list()


def tokenize_clean(text):
    if (text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word
                  in nltk.word_tokenize(sent)]
        # clean token from numeric and other character like puntuation
        filtered_tokens = []
        for token in tokens:
            txt = re.findall('[a-zA-Z]{3,}', token)
            if txt:
                filtered_tokens.append(txt[0])
        return filtered_tokens


def remove_stopwords(tokenized_text):
    if (tokenized_text):
        cleaned_token = []
        for token in tokenized_text:
            if token not in id_stopword:
                cleaned_token.append(token)

        return cleaned_token


def stem_text(tokenized_text):
    if (tokenized_text):
        stems = []
        for token in tokenized_text:
            stems.append(stemmer.stem(token))

        return stems


def remove_en_stopwords(text):
    if text:
        return [token for token in text if token not in en_stopwords]


def stem_en_text(text):
    if text:
        return [en_stemmer.stem(word) for word in text]


def revome_slash_n(text):
    if text:
        return [str(txt).replace("\n", " ") for txt in text]


def lower_text(text):
    if text:
        return [str(txt).lower() for txt in text]


def make_sentence(arr):
    if arr:
        return " ".join(arr)


def text_preprocessing(text):
    if text:
        prep01 = tokenize_clean(text)
        prep02 = remove_stopwords(prep01)
        prep03 = stem_text(prep02)
        prep04 = remove_en_stopwords(prep03)
        prep05 = stem_en_text(prep04)
        prep06 = revome_slash_n(prep05)
        prep07 = lower_text(prep06)
        prep08 = make_sentence(prep07)
        return prep08


tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))
tfidf_model_fp = "model/tfdif_vectorizer.pkl"
with open(tfidf_model_fp, "rb") as fi:
    tfidf_vectorizer = pickle.load(fi)
    print("tfidf loaded from ", tfidf_model_fp)

tfidf_model_fp_enriched = "model/tfdif_vectorizer_enriched.pkl"
with open(tfidf_model_fp_enriched, "rb") as fi:
    tfidf_vectorizer_enriched = pickle.load(fi)
    print("tfidf_enriched loaded from ", tfidf_model_fp_enriched)

tfidf_block_model_fp = "model/tfidf_block.pkl"
with open(tfidf_block_model_fp, "rb") as fi:
    tfidf_vectorizer_block = pickle.load(fi)
    print("tfidf_block loaded from ", tfidf_block_model_fp)

knn_model_path = "model/knn.pkl"
knn_index_path = "model/knn_idx.pkl"
with open(knn_model_path, "rb") as fi:
    knn = pickle.load(fi)
    print("KNN loaded from ", knn_model_path)
with open(knn_index_path, "rb") as fi:
    knn_index = pickle.load(fi)
    print("Index KNN loaded from ", knn_index_path)

knn_model_path_enriched = "model/knn_enriched.pkl"
knn_index_path_enriched = "model/knn_idx_enriched.pkl"
with open(knn_model_path_enriched, "rb") as fi:
    knn_enriched = pickle.load(fi)
    print("KNN loaded from ", knn_model_path_enriched)
with open(knn_index_path_enriched, "rb") as fi:
    knn_index_enriched = pickle.load(fi)
    print("Index KNN loaded from ", knn_index_path_enriched)

knn_block_index_path = "model/knn_block_idx.pkl"
with open(knn_block_index_path, "rb") as fi:
    knn_block_index = pickle.load(fi)
    print("KNN index loaded from ", knn_block_index_path)
knn_block_path = "model/knn_block.pkl"
with open(knn_block_path, "rb") as fi:
    knn_block = pickle.load(fi)
    print("KNN loaded from ", knn_block_path)


def vectorize_tfidf(txt):
    densematrix = tfidf_vectorizer.transform([txt])
    skillvecs = densematrix.toarray().tolist()
    vector = np.array(skillvecs[0]).astype('float32').tolist()
    return vector


def vectorize_tfidf_enriched(txt):
    densematrix = tfidf_vectorizer_enriched.transform([txt])
    skillvecs = densematrix.toarray().tolist()
    vector = np.array(skillvecs[0]).astype('float32').tolist()
    return vector


def vectorize_tfidf_block(txt):
    densematrix = tfidf_vectorizer_block.transform([txt])
    skillvecs = densematrix.toarray().tolist()
    vector = np.array(skillvecs[0]).astype('float32').tolist()
    return vector


df_dupak_all = pd.read_csv("data/dupak_all.csv", sep=";")

kamus = {}
with open("data/dict.json") as fi:
    kamus = json.load(fi)

kamus_normalized = {}
with open("data/kamus_normalized.json") as fi:
    kamus_normalized = json.load(fi)


def enrich_activity(txt):
    txt = str(txt).lower()
    arr = re.findall("\w+", txt)
    final_result = " ".join(arr)
    template = final_result
    for token in arr:
        syns = []
        try:
            syns = kamus[token]["sinonim"]
        except:
            try:
                syns = kamus_normalized[token]["sinonim"]
            except:
                pass
        for syn in syns:
            if syn is not None:
                a = str(template).replace(token, syn)
                final_result += ". " + a
    return final_result


class Response_Item(BaseModel):
    code: str
    activity: str
    level: str
    credit: float


class Responses(BaseModel):
    val: Dict[str, Response_Item]


docx_dict = {}
template = "data/TEMPLATE2.docx"
session_id = ""

# red = redis.Redis(
#     host='127.0.0.1',
#     port=6379,
#     password=''
# )


def ge_activity_code(txt):
    arr = re.findall(
        "[a-z0-9]{1,2}\.[a-z0-9]{1,2}\.*[a-z0-9]*\.*[a-z0-9]*", str(txt).lower())
    if arr:
        return arr[0]
    else:
        return []


EMPTY_STR = "empty"
EMPTY_TH = 0.99999





class Query(BaseModel):
    q: str

headers={'Access-Control-Allow-Origin': '*',
'Access-Control-Allow-Methods': '*',
'Access-Control-Allow-Headers': '*'
}

@app.post("/test/")
async def test(request: Request):
    return {"response": await request.headers.get('Content-Type')}


@app.post("/search/")
# async def search(q: str=Body(embed=True)):
async def search(request: Request):
    result = {}
    # form = await request.form()
    form = await request.json()
    q = str(form["q"])
    # q = str(query["q"])
    if q:
        merge_result = {"vanilla": {}, "enriched": {}}
        skipped_idx_queue = queue.Queue()
        skipped_distance_queue = queue.Queue()
        q_cleansed = text_preprocessing(q)
        if not q_cleansed.strip():
            q_cleansed = EMPTY_STR
        # print("===============================q_cleansed:{}============================".format(q_cleansed))
        q_vector = vectorize_tfidf(q_cleansed)
        (v_distances, v_indices) = knn.kneighbors([q_vector], n_neighbors=5)
        v_indices = v_indices.tolist()
        res = [knn_index[x] for x in v_indices[0]]
        for i, x in enumerate(res):
            merge_result["vanilla"][i] = {"index": v_indices[0][i], "distance": v_distances[0][i],
                                          "activity": x, "ak": df_dupak_all.loc[v_indices[0][i], ["ak"]][0]}

        # q2=enrich_activity(q)
        # q_cleansed=text_preprocessing(q2)
        q_vector = vectorize_tfidf_enriched(q_cleansed)
        (e_distances, e_indices) = knn_enriched.kneighbors(
            [q_vector], n_neighbors=5)
        e_indices = e_indices.tolist()
        res = [knn_index_enriched[x] for x in e_indices[0]]
        for i, x in enumerate(res):
            merge_result["enriched"][i] = {"index": e_indices[0][i], "distance": e_distances[0][i], "activity": x,
                                           "ak": df_dupak_all.loc[e_indices[0][i], ["ak"]][0]}
        elapsed_idx = []
        skipped_idx = []

        vanilla_index = merge_result["vanilla"]
        total_idx = v_indices[0] + e_indices[0]
        # intersected_index=list(set(v_indices[0]).intersection(e_indices[0]))
        elapsed_idx = []
        skipped_idx = []
        result = {}

        # counter=1
        result_num = 5
        for i in range(5):
            tmp = {}
            d = 99
            if merge_result["vanilla"][i]["distance"] < merge_result["enriched"][i]["distance"]:
                idx = merge_result["vanilla"][i]["index"]
                if idx not in elapsed_idx:
                    if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                        skipped_idx = merge_result["enriched"][i]["index"]
                        skipped_d = merge_result["enriched"][i]["distance"]

                    else:
                        skipped_idx = -1
                    tmp["distance"] = merge_result["vanilla"][i]["distance"]
                else:
                    if not skipped_idx_queue.empty():
                        idx = skipped_idx_queue.get()
                        d = skipped_distance_queue.get()
                        tmp["distance"] = d
                tmp["model"] = "vanilla"
            else:
                idx = merge_result["enriched"][i]["index"]
                if idx not in elapsed_idx:
                    if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                        skipped_idx = merge_result["vanilla"][i]["index"]
                        skipped_d = merge_result["vanilla"][i]["distance"]

                    else:
                        skipped_idx = -1
                    tmp["distance"] = merge_result["enriched"][i]["distance"]
                else:
                    if not skipped_idx_queue.empty():
                        idx = skipped_idx_queue.get()
                        d = skipped_distance_queue.get()
                        tmp["distance"] = d
                tmp["model"] = "enriched"

            tmp["index"] = idx
            tmp["code"] = df_dupak_all.loc[idx, "activity_code"]
            tmp["activity"] = df_dupak_all.loc[idx, "activities"]
            tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
            tmp["credit"] = df_dupak_all.loc[idx, "ak"]
            # tmp["session"]=get_docx_link(idx)
            result[i] = tmp
            elapsed_idx.append(idx)
            if skipped_idx != -1:
                skipped_idx_queue.put(skipped_idx)
                skipped_distance_queue.put(skipped_d)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data,headers=headers)


@app.post("/search2/")
async def search2(request: Request):
    result = {}
    # form = await request.form()
    form = await request.json()
    q = str(form["q"])
    # q = str(query["q"])
    if q:
        result = {}
        merge_result = {"vanilla": {}, "enriched": {}, "block": {}}
        skipped_idx_queue = queue.Queue()
        skipped_distance_queue = queue.Queue()
        activitycode = ge_activity_code(q)
        if activitycode:
            actvity_code_only = activitycode
            dftmp = df_dupak_all.query("actvity_code_only=='{}'".format(actvity_code_only))
            for a, i in enumerate(dftmp.index):
                tmp = {}
                try:
                    tmp["distance"] = 0.0
                    tmp["model"] = "code_search"
                    tmp["index"] = str(
                        i)+"_"+df_dupak_all.loc[i, "activity_code"]
                    tmp["code"] = df_dupak_all.loc[i, "actvity_code_only"]
                    tmp["activity"] = df_dupak_all.loc[i, "activity_last_part"]
                    tmp["activity_full"] = df_dupak_all.loc[i, "activities"]
                    tmp["level"] = df_dupak_all.loc[i, "jenjang"]
                    tmp["credit"] = float(df_dupak_all.loc[i, "ak"])
                except:
                    tmp["distance"] = 0.0
                    tmp["model"] = "-"
                    tmp["index"] = "-"
                    tmp["code"] = "-"
                    tmp["activity"] = "-"
                    tmp["activity_full"] = '-'
                    tmp["level"] = "-"
                    tmp["credit"] = 0.0
                result[str(a)] = tmp
        else:
            try:
                q_cleansed = text_preprocessing(q)
            except:
                q_cleansed = EMPTY_STR

            if not q_cleansed:
                q_cleansed = EMPTY_STR
            q_vector = vectorize_tfidf(q_cleansed)
            (v_distances, v_indices) = knn.kneighbors(
                [q_vector], n_neighbors=5)
            v_indices = v_indices.tolist()
            res = [knn_index[x] for x in v_indices[0]]
            for i, x in enumerate(res):
                merge_result["vanilla"][i] = {"index": v_indices[0][i],
                                              "distance": v_distances[0][i],
                                              "activity": x,
                                              "activity_full": df_dupak_all.loc[v_indices[0][i], ["activities"]][0],
                                              "ak": df_dupak_all.loc[v_indices[0][i], ["ak"]][0]}

            # q2=enrich_activity(q)
            # q_cleansed=text_preprocessing(q2)
            q_vector = vectorize_tfidf_enriched(q_cleansed)
            (e_distances, e_indices) = knn_enriched.kneighbors(
                [q_vector], n_neighbors=5)
            e_indices = e_indices.tolist()
            res = [knn_index_enriched[x] for x in e_indices[0]]
            for i, x in enumerate(res):
                merge_result["enriched"][i] = {"index": e_indices[0][i], 
                "distance": e_distances[0][i], 
                "activity": x,
                "activity_full": df_dupak_all.loc[e_indices[0][i], ["activities"]][0],
                "ak": df_dupak_all.loc[e_indices[0][i], ["ak"]][0]}

            # q_vector3 = vectorize_tfidf_block(q_cleansed)
            # (e_distances3, e_indices3) = knn_block.kneighbors([q_vector3], n_neighbors=5)
            # e_indices3 = e_indices3.tolist()
            # res = [knn_block_index[x] for x in e_indices3[0]]
            # for i, x in enumerate(res):
            #     merge_result["block"][i] = {"index": e_indices3[0][i], "distance": e_distances3[0][i], "activity": x,
            #                                    "ak": df_dupak_all.loc[e_indices3[0][i], ["ak"]][0]}

            vanilla_index = merge_result["vanilla"]
            total_idx = v_indices[0] + e_indices[0]
            intersected_index = list(
                set(v_indices[0]).intersection(e_indices[0]))
            elapsed_idx = []
            skipped_idx = []
            result = {}

            # counter=1
            result_num = 5
            for i in range(5):
                tmp = {}
                d = 99
                if merge_result["vanilla"][i]["distance"] < merge_result["enriched"][i]["distance"]:
                    idx = merge_result["vanilla"][i]["index"]
                    if idx not in elapsed_idx:
                        if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                            skipped_idx = merge_result["enriched"][i]["index"]
                            skipped_d = merge_result["enriched"][i]["distance"]

                        else:
                            skipped_idx = -1
                        tmp["distance"] = merge_result["vanilla"][i]["distance"]
                    else:
                        if not skipped_idx_queue.empty():
                            idx = skipped_idx_queue.get()
                            d = skipped_distance_queue.get()
                            tmp["distance"] = d
                    tmp["model"] = "vanilla"
                else:
                    idx = merge_result["enriched"][i]["index"]
                    if idx not in elapsed_idx:
                        if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                            skipped_idx = merge_result["vanilla"][i]["index"]
                            skipped_d = merge_result["vanilla"][i]["distance"]

                        else:
                            skipped_idx = -1
                        tmp["distance"] = merge_result["enriched"][i]["distance"]
                    else:
                        if not skipped_idx_queue.empty():
                            idx = skipped_idx_queue.get()
                            d = skipped_distance_queue.get()
                            tmp["distance"] = d
                    tmp["model"] = "enriched"

                c = df_dupak_all.loc[idx, "activity_code"]
                tmp["index"] = str(idx)+"_"+c
                tmp["code"] = c
                tmp["activity"] = df_dupak_all.loc[idx, "activity_last_part"]
                tmp["activity_full"] = df_dupak_all.loc[idx, "activities"]
                tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
                try:
                    tmp["credit"] = float(df_dupak_all.loc[idx, "ak"])
                except:
                    tmp["credit"] = df_dupak_all.loc[idx, "ak"]
                
                # tmp["session"]=get_docx_link(idx)
                result[i] = tmp
                elapsed_idx.append(idx)
                if skipped_idx != -1:
                    skipped_idx_queue.put(skipped_idx)
                    skipped_distance_queue.put(skipped_d)

    arr = []
    meanval = math.floor(
        (np.mean([v["distance"] for k, v in result.items()])*100000))/100000.00
    if meanval != EMPTY_TH:
        for i, res in result.items():
            arr.append(res)
    result2 = {"results": arr}
    json_compatible_item_data = jsonable_encoder(result2)
    # json_compatible_item_data = jsonable_encoder(final)
    return JSONResponse(content=json_compatible_item_data,headers=headers)


@app.post("/download/", response_description='docx')
async def download(request: Request):
    result = {}
    # form = await request.form()
    form = await request.json()
    idx = str(form["idx"])
    act = str(form["act"])
    # q = str(query["q"])
    if idx:
        d = datetime.now().date()
        tpl = DocxTemplate(template)
        idx = int(idx)
        context = {}
        # context["nip_prakom"] = "198401112009011004"
        # context["nip_atasan"] = "198401112009011005"
        ac = df_dupak_all.at[idx, "activity_code"]
        arrs = ac.split("_")
        jenjang = arrs[0]
        cd = arrs[1]
        doc_title = cd+"_"+act
        evdnt = eval(df_dupak_all.at[idx, "evidents"])
        evdnt = [str(i)+") "+x for i, x in enumerate(evdnt) if x]
        # print("doc_title:{}".format(doc_title))
        context["kode_kegiatan"] = cd
        context["bukti_kegiatan"] = "\n\n\n".join(evdnt)
        context["judul_kegiatan"] = df_dupak_all.at[idx, "activity_last_part"]
        context["lokasi"] = "Jakarta"
        context["query_kegiatan"] = act
        context["keterangan_kegiatan"] = "-"
        context["nama_atasan"] = "@nama_atasan"
        context["nama_prakom"] = "@nama_prakom"
        context["pangkat"] = "@pangkat_prakom"
        context["jenjang_prakom"] = jenjang
        context["tanggal"] = str(d.day)+"-"+str(d.month)+"-"+str(d.year)
        context["angka_kredit"] = df_dupak_all.at[idx, "ak"]
        context["golongan"] = "@golongan_prakom"
        tpl.render(context)
        # tpl.save("doctpl.docx")

        # Create in-memory buffer
        file_stream = io.BytesIO()
        # Save the .docx to the buffer
        tpl.save(file_stream)
        # Reset the buffer's file-pointer to the beginning of the file
        file_stream.seek(0)
        doc_title = doc_title.replace(" ", "_")+".docx"
        thisheaders = {
            'Content-Disposition': "'attachment; filename="+doc_title
        }
        return StreamingResponse(file_stream, headers=thisheaders)


uri = "bolt://"+config('NEO4J_HOST')+":"+config('NEO4J_PORT')
user = config('NEO4J_USER')
password = config('NEO4J_PASS')
neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))


def text_preprocessing_id(text):
    if text:
        prep01 = tokenize_clean(text)
        prep02 = remove_stopwords(prep01)
        prep03 = stem_text(prep02)
        prep06 = revome_slash_n(prep03)
        prep07 = lower_text(prep06)
        prep08 = make_sentence(prep07)
        return prep08


def text_preprocessing_en(text):
    if text:
        prep01 = tokenize_clean(text)
        prep04 = remove_en_stopwords(prep01)
        prep05 = stem_en_text(prep04)
        prep06 = revome_slash_n(prep05)
        prep07 = lower_text(prep06)
        prep08 = make_sentence(prep07)
        return prep08


def get_word(txt, ref, preserve_empty_words=False):
    word = ""
    try:
        rel = txt.split()
        rels = []
        for r in rel:
            a = r.split("_")
            if len(a) == 2:
                pos = a[0]
                idx = a[-1]
                label = ref[int(idx)]
                if pos in ["FW"]:  # ["FW","Z"]:
                    label2 = text_preprocessing_en(label)
                else:
                    label2 = text_preprocessing_id(label)
                if label2:
                    rels.append(label2)
                else:
                    if preserve_empty_words:
                        rels.append(label)
        word = " ".join(rels)
    except Exception as x:
        print("ERROR on text {} why?:{}".format(txt, str(x)))
    return word


def get_graph_rel(txt):
    tagged0 = tagger.tag_sents([txt.split()])
    if tagged0[0][0][-1] == 'VB':
        txt = 'Pegawai '+txt
        tagged0 = tagger.tag_sents([txt.split()])
    ori_pos = [pos[-1]+"_"+str(i) for i, pos in enumerate(tagged0[0])]
    ori_word = [pos[0] for i, pos in enumerate(tagged0[0])]
    pos_sent = " ".join(ori_pos)
    nnps = re.findall(
        "[NPFWZ_\d]{4,6}\s[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*[NPFWZ_\d]*\s*", pos_sent)
    prev = ""
    edges = []
    for nnp in nnps:
        nnp = str(nnp).strip()
        if prev:
            rel = re.findall("(?<="+prev+").*?(?="+nnp+")", pos_sent)
#             print("REL:",rel)
            if rel:
                rel = rel[0]
                try:
                    #                     i=int(rel.split("_")[-1])
                    # ori_word[i]
                    vb = get_word(rel, ori_word, preserve_empty_words=True)
#                     print(rel+'++++++++'+vb)
                    if re.findall("\w+", vb):
                        r = vb  # get_word(rel,ori_word)
                        nn1 = get_word(prev, ori_word)
                        nn2 = get_word(nnp, ori_word)
                        if nn1 and nn2:
                            edges.append([nn1, r, nn2, 0.55, 147])
                except Exception as x:
                    print("ERR on vb:{}:{}".format(rel, x))
                    with open('data/err.txt', 'a+') as fo:
                        fo.write(prev+'---'+rel+'---'+nnp+'\n')

        prev = nnp

    return edges


@app.post("/query_graph/")
async def query_graph(request: Request):
    result = {}
    form = await request.json()
    q = str(form["q"])
    if q:
        arr = []
        with neo4j_driver.session() as session:
            for x in get_graph_rel(q):
                result = session.run("match p=(m)-[r]-(n) where toLower(m.label) contains '"+str(x[0]).lower()+"' \
                                 or toLower(n.label) contains '"+str(x[2]).lower()+"' return m,r,n")
                tmp = {'node1': {}, 'rel': {}, 'node2': {}}
                for record in result:
                    tmp['node1']['label'] = record['m']['label']
                    tmp['rel']['label'] = record['r']['label']
                    tmp['rel']['idx'] = record['r']['idx']
                    tmp['rel']['score'] = record['r']['score']
                    tmp['node2']['label'] = record['n']['label']
                    arr.append(tmp)
        result = {"results": arr}
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data,headers=headers)

def rgb_to_hex(rgb):
  r,g,b=rgb
  return '#%02x%02x%02x' % (r,g,b)

def get_bright_color():
    h,s,l = random.random(), 0.35 - random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return rgb_to_hex((r, g, b))

@app.post("/query_graph2/")
async def query_graph2(request: Request):
    result = {}
    form = await request.json()
    q = str(form["q"])
    if q:
        arr = []
        with neo4j_driver.session() as session:
            nodes= []
            edges= []
            tmp_ids=[]
            for x in get_graph_rel(q):
                result = session.run("match p=(m)-[r]-(n) where toLower(m.label) contains '"+str(x[0]).lower()+"' \
                                 or toLower(n.label) contains '"+str(x[2]).lower()+"' return m,r,n")
                tmp_edges=[]
                for i,record in enumerate(result):
                    name1=str(record['m']['label']).replace(' ','\n')
                    name2=str(record['n']['label']).replace(' ','\n')
                    indx=record['r']['idx']
                    scor=record['r']['score']
                    lbl=record['n']['label']
                    idx_scor='_'.join([str(scor),str(indx)])
                    # n2m=str(record['m']['label'])+'->'+str(record['n']['label'])
                    # if n2m not in tmp_edges:
                    edge_id='_'.join([str(uuid.uuid1()),idx_scor])
                    edges.append({ 'id':edge_id,'from': name1, 'to': name2 ,'label':lbl})
                    if name1 not in tmp_ids:
                        nodes.append({ 'id': name1, 'label': name1}) #,'color':get_bright_color()})
                        tmp_ids.append(name1)
                    if name2 not in tmp_ids:
                        nodes.append({ 'id': name2, 'label': name2}) #,'color':get_bright_color() })
                        tmp_ids.append(name2)
                        # tmp_edges.append(n2m)
                arr={'nodes':nodes,'edges':edges}
        result = {"results": arr}
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data,headers=headers)


@app.post("/get_dupak_by_index/")
async def get_dupak_by_index(request: Request):
    result = {}
    # form = await request.form()
    form = await request.json()
    idx = str(form["q"])
    tmp={}
    tmp["distance"] = 0.0
    tmp["model"] = "-"
    tmp["index"] = "-"
    tmp["code"] = "-"
    tmp["activity"] = "-"
    tmp["activity_full"] = '-'
    tmp["level"] = "-"
    tmp["credit"] = 0.0
    if idx:
        idx=int(idx)
        tmp = {}
        try:
            tmp["distance"] = 0.0
            tmp["model"] = "code_search"
            tmp["index"] = idx
            tmp["code"] = df_dupak_all.loc[idx, "actvity_code_only"]
            tmp["activity"] = df_dupak_all.loc[idx, "activity_last_part"]
            tmp["activity_full"] = df_dupak_all.loc[idx, "activities"]
            tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
            tmp["credit"] = float(df_dupak_all.loc[idx, "ak"])
        except:
            pass
    result2 = {"results": [tmp]}
    json_compatible_item_data = jsonable_encoder(result2)
    return JSONResponse(content=json_compatible_item_data,headers=headers)


@app.post("/get_dupak_by_indexes/")
async def get_dupak_by_indexes(request: Request):
    # form = await request.form()
    form = await request.json()
    idxes = str(form["q"])
    idxes=idxes.split(';')
    idxes=list(set(idxes))
    arr=[]
    for idx in idxes:
        try:
            idx=int(idx)
            tmp = {}
            try:
                tmp["distance"] = 0.0
                tmp["model"] = "code_search"
                tmp["index"] = str(idx)+"_"+df_dupak_all.loc[idx, "activity_code"]
                tmp["code"] = df_dupak_all.loc[idx, "actvity_code_only"]
                tmp["activity"] = df_dupak_all.loc[idx, "activity_last_part"]
                tmp["activity_full"] = df_dupak_all.loc[idx, "activities"]
                tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
                tmp["credit"] = float(df_dupak_all.loc[idx, "ak"])
                arr.append(tmp)
            except:
                tmp["distance"] = 0.0
                tmp["model"] = "-"
                tmp["index"] = "-"
                tmp["code"] = "-"
                tmp["activity"] = "-"
                tmp["activity_full"] = '-'
                tmp["level"] = "-"
                tmp["credit"] = 0.0
                arr.append(tmp)
        except:
            tmp = {}
            tmp["distance"] = 0.0
            tmp["model"] = "-"
            tmp["index"] = "-"
            tmp["code"] = "-"
            tmp["activity"] = "-"
            tmp["activity_full"] = '-'
            tmp["level"] = "-"
            tmp["credit"] = 0.0
            arr.append(tmp)

    result2 = {"results": arr}
    json_compatible_item_data = jsonable_encoder(result2)
    return JSONResponse(content=json_compatible_item_data,headers=headers)

from fastapi import FastAPI, File
from starlette.requests import Request
import io
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import tensorflow as tf
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
import src.facenet as facenet
import uvicorn
import cv2
from scipy import spatial
import pymongo
import numpy as np
import json
from confluent_kafka import Producer, Consumer, KafkaError

p = Producer({'bootstrap.servers': '10.235.4.134:9092,10.235.4.137:9092,10.235.4.146:9092,10.235.4.147:9092,10.235.4.148:9092'})

connection = pymongo.MongoClient('mongodb://app_it_facerecog:f4c3IT2021##@dc01-mongo.kemenkeu.go.id:27017/',authSource='it_facerecog')
db_facenet = connection.it_facerecog
col_average_vector=db_facenet.col_average_vector

detector_model=r"../../corpus/facenet/weight/ultra_light_320.onnx"
facenet_model =r"../../corpus/facenet/weight/20180402-114759.pb"
model_folder=r"../../corpus/facenet/models/average/ue2_staffs_and_leaders"
nip_id_name_org_pkl=r"../../corpus/facenet/models/nip_id_name_org.pkl"
id_nip_name_org_pkl=r"../../corpus/facenet/models/id_nip_name_org.pkl"
dev_path = r"../../corpus/facenet/dev"
error_log=r"../../corpus/facenet/errors/15_errors.txt"

onnx_model = onnx.load(detector_model)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(detector_model)
input_name = ort_session.get_inputs()[0].name
onnx_dim=(320,240)

def detect_face_using_onnx(img_bgr):
    result=(None,[0,0,0,0])
    img_onnx_input = cv2.resize(img_bgr, onnx_dim)
    img_onnx_input_rgb = cv2.cvtColor(img_onnx_input, cv2.COLOR_BGR2RGB)
    img_mean = np.array([127, 127, 127])
    img = (img_onnx_input_rgb - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    confidences, boxes = ort_session.run(None, {input_name: img})
    w, h = onnx_dim[0], onnx_dim[1]
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        x1=x1 if x1>=0 else 0
        y1 = y1 if y1 >= 0 else 0
        x2 = x2 if x2 >= 0 else 0
        y2 = y2 if y2 >= 0 else 0
        break
    if boxes.shape[0] > 0:
        face_rgb = img_onnx_input_rgb[y1:y2, x1:x2]
        shape = np.shape(face_rgb)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(face_bgr, (shape[0] + 13, shape[1]), interpolation=cv2.INTER_AREA)  ####convert from landscape to portray face shape
        result = (resized, [x1, y1, x2, y2])
    return result

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

with open(facenet_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
graph.finalize()
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
)
sess = tf.Session(graph=graph,
                  config=sess_config)
np.random.seed(seed=1)
# Load the model
print('Loading feature extraction model')
facenet.load_model(facenet_model)

test_img_num = 1
images_placeholder = graph.get_tensor_by_name("input:0")  # tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")  # tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")  # tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
emb_array_using_io = np.zeros((test_img_num, embedding_size)).astype("float32")
emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
nearest_neighbor = 1
facenet_dim=(160, 160)
T=0.10   #0.65 #0.5 #0.76
arr=col_average_vector.find()
average_unmasked_model={}
for x in arr:
    average_unmasked_model[x['id_hris']]=x['average_vector']  #convert "[]" from mngodb to [] in python

class Response_Item(BaseModel):
    status: str
    timestamp: datetime.datetime
    prediction: str
    elapsed: float

class Registration_Response(BaseModel):
    status: str
    timestamp: datetime.datetime
    message: str

class Verify_Response_Item(BaseModel):
    status: str
    timestamp: datetime.datetime
    prediction: str
    elapsed: float
    score: float

app = FastAPI()
@app.post("/register/")
async def verify(request: Request,
                 face: bytes = File(...)):
    emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
    is_verified = False
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    # result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if id in average_unmasked_model:
                result.status = "User already exist"
                result.message = ""
            # elif col_average_vector.find_one({"id_hris":id}):
            #     result.status = "User already exist"
            #     result.message = ""
            else:
                # absent_date=form["absent_date"]     #format '2018-06-29 08:15:27.243860'
                stream = io.BytesIO(face)
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                # (face_bgr, [x1, y1, x2, y2]) = detect_face_using_onnx(img_bgr)
                (face_bgr, _) = detect_face_using_onnx(img_bgr)
                if face_bgr is not None:
                    facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                    facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                    facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                    nrof_samples = 1
                    imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                    imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                    feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                    emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array_using_cv2=np.reshape(emb_array_using_cv2,-1)
                    vec=[float(x) for x in emb_array_using_cv2]
                    col_average_vector.insert_one({"id_hris":int(id),"average_vector":vec})  #insert to DB
                    average_unmasked_model[id] = vec                                         # insert to memory
                    p.poll(0)
                    data = json.dumps({"id": id, "vec": vec})
                    p.produce('facecreate', key="facecreate", value=data.encode('utf-8'), callback=delivery_report)
                    p.flush()
                    result.timestamp = datetime.datetime.now()#strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                    result.status = "ok"
                    result.message="User {} berhasil didaftarkan".format(str(id))
                else:
                    result.status = "Face NOT found"
                    result.message = "Face NOT found"

        except Exception as e:
            result.status = "Error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/verify/")
async def verify(request: Request,
                 face: bytes = File(...)):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        'prediction': "-1",
        "elapsed": 0.0,
        "score":0.0
    }
    is_verified = False
    result = Verify_Response_Item(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            # absent_date=form["absent_date"]     #format '2018-06-29 08:15:27.243860'
            stream = io.BytesIO(face)
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            (face_bgr, _ ) = detect_face_using_onnx(img_bgr)
            avg_vec=[]
            if face_bgr is not None:
                facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                nrof_samples = 1
                imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                try:
                    avg_vec = average_unmasked_model[id]
                except Exception as IDNotFoundException:
                    doc = col_average_vector.find_one({"id_hris": id})
                    if doc:
                        avg_vec=doc['average_vector']
                        average_unmasked_model[id]=avg_vec  ###update recen memory based on DB
                    else:
                        result.status = "Error"
                        # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                        elapsed = (datetime.datetime.now() - t0).total_seconds()
                        result.elapsed = elapsed
                        result.prediction = "User NOT found"
                sim_score = 1 - spatial.distance.cosine(avg_vec, emb_array_using_cv2)
                if sim_score > T:
                    is_verified = True
                else:
                    is_verified = False
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                result.status = "OK"
                # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                result.elapsed = elapsed
                result.prediction = "verified" if is_verified else "not verified"
                result.score=sim_score
            else:
                result.status = "Error"
                # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                result.elapsed = elapsed
                result.prediction = "Face NOT found"
        except Exception as e:
            result.status = "error:%s"%str(e)
            result.timestamp = datetime.datetime.now()
            print("ERROR SERVER:%s" % str(e))

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class Response_Item2(BaseModel):
    status: str
    is_exist:bool

@app.post("/exist/")
async def exist(request: Request):
    init_result2 = {
        'status': 'Ok',
        'is_exist': False
    }
    is_exist = False
    result = Response_Item2(**init_result2)
    if request.method == "POST":
        try:
            form = await request.form()
            id=int(form["id"])
            try:
                avg_vec = average_unmasked_model[id]
                result.status = "Ok"
                result.is_exist = True
            except Exception as IDNotFoundException:
                doc = col_average_vector.find_one({"id_hris": id})
                if doc:
                    avg_vec=doc['average_vector']
                    average_unmasked_model[id]=avg_vec  ###update recen memory based on DB
                    result.status = "Ok"
                    result.is_exist = True
                else:
                    result.status = "Ok"
                    result.is_exist = False
        except Exception as e:
            result.status = "error:%s"%str(e)
            result.is_exist = False
            print("ERROR SERVER:%s" % str(e))

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

@app.post("/update/")
async def update(request: Request,
                 face: bytes = File(...)):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
    is_verified = False
    result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if (id in average_unmasked_model) or (col_average_vector.find_one({"id_hris":id})):
                stream = io.BytesIO(face)
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                (face_bgr, _) = detect_face_using_onnx(img_bgr)
                if face_bgr is not None:
                    facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                    facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                    facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                    nrof_samples = 1
                    imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                    imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                    feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                    emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array_using_cv2 = np.reshape(emb_array_using_cv2, -1)
                    vec = [float(x) for x in emb_array_using_cv2]
                    col_average_vector.update_one({"id_hris": int(id)},{'$set': {'average_vector': vec}}, upsert=False)
                    average_unmasked_model[id] = vec  # update to memory
                    p.poll(0)
                    data=json.dumps({"id":id,"vec":vec})
                    p.produce('faceupdate', key="faceupdate",value=data.encode('utf-8'), callback=delivery_report)
                    p.flush()
                    result.timestamp = datetime.datetime.now()  # strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                    result.status = "ok"
                    result.message = "Face vector {} berhasil diupdate".format(str(id))
                else:
                    result.status = "error"
                    result.message = "Face NOT found"
            else:
                result.status = "error"
                result.message = "User Not Found"


        except Exception as e:
            result.status = "error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/delete/")
async def delete(request: Request):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if average_unmasked_model[id]:
                del average_unmasked_model[id]
                col_average_vector.delete_many({"id_hris": int(id)})
                p.poll(0)
                data=json.dumps({"id":id})
                p.produce('facedelete', key="facedelete",value=data.encode('utf-8'), callback=delivery_report)
                p.flush()
                result.timestamp = datetime.datetime.now()  # strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                result.status = "ok"
                result.message = "Face vector {} berhasil didelete".format(str(id))
            else:
                result.status = "error"
                result.message = "User Not Found"
        except Exception as e:
            result.status = "error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class Item(BaseModel):
    id: int
    vec: list

@app.post("/create_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        vec=item.vec
        average_unmasked_model[id] = vec
        result.status="ok"
        result.message="cache facevector {} is created".format(id)
        result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/update_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        vec=item.vec
        average_unmasked_model[id] = vec
        result.status="ok"
        result.message="cache facevector {} is updated".format(id)
        result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/delete_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        if average_unmasked_model[id]:
            del average_unmasked_model[id]
            result.status="ok"
            result.message="cache facevector {} is deleted".format(id)
            result.timestamp = datetime.datetime.now()
        else:
            result.status = "ok"
            result.message = "user not found"
            result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # , reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
