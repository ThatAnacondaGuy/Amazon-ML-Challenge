import sys, collections
for path in sys.argv[1:]:
    nf = collections.Counter(); n=0; dq=0; sq=0; cr=0; bad_utf8=0; blank=0; lead_trail_ws=0; ctrl=0
    with open(path,'rb') as f:
        header=f.readline()
        for raw in f:
            n+=1
            if b'\r' in raw: cr+=1
            try: line=raw.decode('utf-8')
            except UnicodeDecodeError: bad_utf8+=1; line=raw.decode('utf-8','replace')
            line=line.rstrip('\n')
            if not line.strip(): blank+=1; continue
            parts=line.split('\t'); nf[len(parts)]+=1
            if '"' in line: dq+=1
            if any(p!=p.strip() for p in parts): lead_trail_ws+=1
            if any(ord(c)<32 and c!='\t' for c in line): ctrl+=1
    print(f"{path}: header={header!r}\n  rows={n:,} fields={dict(nf)} dquote_rows={dq:,} CR_rows={cr} bad_utf8={bad_utf8} blank={blank} ws_padded_rows={lead_trail_ws:,} ctrl_char_rows={ctrl}")
