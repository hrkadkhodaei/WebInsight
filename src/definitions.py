scale_dict = {
    'contentLength': 'log',
    'textSize': 'symlog',
    'trustRank': 'symlog',
    'numInternalOutLinks': 'symlog',
    'numExternalOutLinks': 'symlog',
    'numInternalInLinks': 'symlog',
    'numExternalInLinks': 'symlog',
    'diffInternalOutLinks-1': 'symlog',
    'diffExternalOutLinks-1': 'symlog',
    'diffInternalOutLinks-2': 'symlog',
    'diffExternalOutLinks-2': 'symlog'
}

feature_label_dict = {
    'contentLength': 'page content size',
    'textSize': 'page text size',
    'numInternalOutLinks': '# internal outlinks',
    'numExternalOutLinks': '# external outlinks',
    'numInternalInLinks': '# internal inlinks',
    'numExternalInLinks': '# external inlinks',
    'textQuality': 'page text quality',
    'pathDepth': 'URL path depth',
    'domainDepth': 'URL domain depth',
    'trustRank': 'TrustRank',
    'diffInternalOutLinks': '# new internal outlinks',
    'diffExternalOutLinks': '# new external outlinks',

    'related_contentLength':       'related page content size',
    'related_textSize':            'related page text size',
    'related_numInternalOutLinks': '# related internal outlinks',
    'related_numExternalOutLinks': '# related external outlinks',
    'related_numInternalInLinks':  '# related internal inlinks',
    'related_numExternalInLinks':  '# related external inlinks',
    'related_textQuality':         'related page text quality',
    'related_pathDepth':           'related URL path depth',
    'related_domainDepth':         'related URL domain depth',
    'related_trustRank':           'related TrustRank',
}

target_label_dict = {
    'diffInternalOutLinks': 'prob. new outlinks (internal)',
    'diffExternalOutLinks': 'prob. new outlinks (external)',
    'changeRate': 'content change rate',
    'linkInternalChangeRate': 'link change rate (internal)',
    'linkExternalChangeRate': 'link change rate (external)'
}

history_label_dict = dict([
    (f + "-" + str(i + 1), feature_label_dict[f] + " (-" + str(i + 1) + ")") for f in feature_label_dict.keys()
    for i in range(8)
])

pretty_label_dict = feature_label_dict.copy()
pretty_label_dict.update(target_label_dict)
pretty_label_dict.update(history_label_dict)

pretty_class_names = {
    0: "0",
    1: "1+"
}

categorical_target_label_dict = {
    'diffInternalOutLinks': 'new outlinks (internal)',
    'diffExternalOutLinks': 'new outlinks (external)'
}

static_page_features = ['contentLength', 'textSize', 'textQuality', 'pathDepth', 'domainDepth', 'numInternalOutLinks',
                        'numExternalOutLinks']  #
static_page_semantics = ["SV" + str(i) for i in range(192)]
static_network_features = ['numInternalInLinks', 'numExternalInLinks', 'trustRank']
static_network_features += ['related_' + f for f in static_page_features + static_network_features]
dynamic_network_features = ['numInternalInLinks-', 'numExternalInLinks-', 'trustRank-']
dynamic_page_features = ['contentLength-', 'textSize-', 'textQuality-', 'diffInternalOutLinks-',
                         'diffExternalOutLinks-']

feature_sets = {
    'SP': static_page_features,
    'v': static_page_semantics,
    'SN': static_network_features,
}

feature_sets.update(dict([
    ('DP' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_page_features])
    for i in range(8)
]))

feature_sets['DPRate'] = ['related_linkExternalChangeRate', 'related_linkInternalChangeRate']

feature_sets.update(dict([
    ('DN' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_network_features])
    for i in range(8)
]))  # 'DP/N8' will contain all -1, ... -8 dynamic page features


