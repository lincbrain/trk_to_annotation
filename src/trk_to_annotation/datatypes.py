SEGMENT_DTYPE = (
    ('streamline', 'i8'),
    ('start', 'f4', 3),
    ('end', 'f4', 3),
    ('orientation', 'f4', 3),
    ('id', 'i8')
)

# Optional per-segment bundle-label fields, appended to SEGMENT_DTYPE when the
# source .trk carries a "label_id" scalar/property (e.g. bundle group id).
# label_name duplicates label_id's raw value under a separate property id so
# Neuroglancer can attach enum_values/enum_labels (bundle names) to it while
# label_id stays a plain numeric property.
LABEL_EXTRA_DTYPE = (
    ('label_id', '<u2'),
    ('label_name', '<u2'),
    ('label_color', '<u1', 3),
)
LABEL_EXTRA_BYTES = 8  # 2 (label_id) + 2 (label_name) + 3 (label_color) + 1 (extra
                        # alignment byte: Neuroglancer requires each record's
                        # byte size to be a multiple of 4; the plain 7 extra
                        # label bytes would break that, so an extra anonymous
                        # 1-byte padding field (LABEL_ALIGN_PAD_FIELD) is
                        # added alongside the normal "padding" field when
                        # labels are present.
LABEL_ALIGN_PAD_FIELD = ("label_align_pad", "u1")  # only when has_label
