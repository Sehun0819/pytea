/*
 * context.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Context for backend processing.
 * Collection of Environment, Heap, and Constraint set.
 */
import { List, Record } from 'immutable';

import { getFileInfo } from 'pyright-internal/analyzer/analyzerNodeInfo';
import { ParseNode, ParseNodeType } from 'pyright-internal/parser/parseNodes';

import { fetchAddr /*genTensor*/ } from './backUtils';
//import { ConstraintSet } from './constraintSet';
/*import {
    Constraint,
    ConstraintType,
    CtrAnd,
    CtrBroad,
    CtrEq,
    CtrFail,
    CtrForall,
    CtrLt,
    CtrLte,
    CtrNeq,
    CtrNot,
    CtrOr,
    ctrToStr,
    extractSymbols,
} from './constraintType';*/
//import { isStructuallyEq, simplifyExp, simplifyShape } from './expUtils';
//import { NumRange } from './range';
import { ShEnv, ShHeap } from './sharpEnvironments';
import {
    CodeSource,
    ShContFlag,
    ShValue,
    SVAddr,
    //SVBool,
    SVError,
    SVErrorLevel,
    //SVFloat,
    SVFunc,
    SVInt,
    //SVNone,
    SVObject,
    //SVSize,
    //SVString,
    //SVType,
} from './sharpValues';
/*import {
    ExpBool,
    ExpNum,
    ExpNumSymbol,
    ExpShape,
    ExpShapeConst,
    NumBopType,
    SymBool,
    SymExp,
    SymFloat,
    SymInt,
    SymShape,
    SymString,
} from './symExpressions';*/
import { Translator } from './torch2hBackend';

let _failId = 0;
function getFailedId(): number {
    return ++_failId;
}
const LOG_IGNORE = /pylib.*(tensor|functional|builtins|linear|module|loss|math).py$/;
function checkIgnorePath(path: string) {
    return LOG_IGNORE.test(path);
}

interface ContextProps<T> {
    failId: number;
    env: ShEnv;
    heap: ShHeap;
    retVal: T;

    // SVFunc is python function call, string is libcall name.
    callStack: List<[SVFunc | string, CodeSource | undefined]>;
    logs: List<ShValue>;
    imported: ShEnv; // qualPath (relative to project root or site-packages) to address.
    relPath: string; // relative path to entry file. starts with entry file name.

    // if set, automatically go to failed path.
    failed?: SVError;
}

interface ContextMethods<T> {
    // property setters.
    setEnv(env: ShEnv): Context<T>;
    setHeap(heap: ShHeap): Context<T>;
    setRetVal<A>(retVal: A): Context<A>;
    setRelPath(relPath: string): Context<T>;
    setImported(imported: ShEnv): Context<T>;

    getAttrDeep(value: ShValue, attr: string, source: CodeSource | undefined): Context<ShValue>;
    getIndiceDeep(value: ShValue, index: number, source: CodeSource | undefined): Context<ShValue>;
    getKeyValDeep(value: ShValue, key: string, source: CodeSource | undefined): Context<ShValue>;

    addLog(message: string, source: CodeSource | undefined): Context<T>;
    addLogValue(log: ShValue): Context<T>;
    pushCallStack(stack: [SVFunc | string, ParseNode | undefined]): Context<T>;
    popCallStack(): Context<T>;

    // these methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError>;
    warnWithMsg(message: string, source: CodeSource | undefined): Context<SVError>;

    // these two methods cut off paths
    fail(error: SVError): Context<SVError>;
    failWithMsg(message: string, source: CodeSource | undefined): Context<SVError>;

    // primitive collections
    genList(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>];
    genTuple(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>];
}

const contextDefaults: ContextProps<unknown> = {
    failId: -1,
    env: new ShEnv(),
    heap: new ShHeap(),
    retVal: undefined,

    callStack: List(),
    logs: List(),
    imported: new ShEnv(),
    relPath: '.',

    failed: undefined,
};

export class Context<T> extends Record(contextDefaults) implements ContextProps<T>, ContextMethods<T> {
    retVal!: T;

    constructor(values?: Partial<ContextProps<T>>) {
        values ? super(values) : super();
    }

    setEnv(env: ShEnv): Context<T> {
        return this.set('env', env);
    }

    setHeap(heap: ShHeap): Context<T> {
        return this.set('heap', heap);
    }

    setRetVal<A>(retVal: A): Context<A> {
        // WARNING: use unknown type hack due to the TProps type parameter of Record
        //          follows type of contextDefaults (ContextProps<unknown>)
        return ((this as unknown) as Context<A>).set('retVal', retVal);
    }

    setRelPath(relPath: string): Context<T> {
        return this.set('relPath', relPath);
    }

    setImported(imported: ShEnv): Context<T> {
        return this.set('imported', imported);
    }

    getAttrDeep(value: ShValue, attr: string, source: CodeSource | undefined): Context<ShValue> {
        return Translator.getAttrDeep(this, value, attr, source);
    }

    getIndiceDeep(value: ShValue, index: number, source: CodeSource | undefined): Context<ShValue> {
        return Translator.getIndiceDeep(this, value, index, source);
    }

    getKeyValDeep(value: ShValue, key: string, source: CodeSource | undefined): Context<ShValue> {
        return Translator.getKeyValDeep(this, value, key, source);
    }

    // these two methods does not cut off paths. just log warnings
    warn(warning: SVError): Context<SVError> {
        return this.setRetVal(warning).addLogValue(warning);
    }
    warnWithMsg(message: string, source: CodeSource | undefined): Context<SVError> {
        source = this._replaceBuiltinSource(source);
        const warning = SVError.create(message, SVErrorLevel.Warning, source);
        return this.setRetVal(warning).addLogValue(warning);
    }

    // these two methods cut off paths
    fail(error: SVError): Context<SVError> {
        return this.set('failed', error).set('failId', getFailedId()).addLogValue(error).setRetVal(error);
    }
    failWithMsg(message: string, source: CodeSource | undefined): Context<SVError> {
        source = this._replaceBuiltinSource(source);
        return this.fail(SVError.create(message, SVErrorLevel.Error, source));
    }

    addLog(message: string, source: CodeSource | undefined): Context<T> {
        source = this._replaceBuiltinSource(source);
        return this.set('logs', this.logs.push(SVError.create(message, SVErrorLevel.Log, source)));
    }

    addLogValue(log: ShValue): Context<T> {
        return this.set('logs', this.logs.push(log));
    }

    pushCallStack(stack: [SVFunc | string, CodeSource | undefined]): Context<T> {
        return this.set('callStack', this.callStack.push(stack));
    }

    popCallStack(): Context<T> {
        return this.set('callStack', this.callStack.pop());
    }

    // shift every addresses to negative.
    asDefault(): Context<T> {
        const offset = -this.heap.addrMax - 1;
        return this.setEnv(this.env.addOffset(offset)).setHeap(this.heap.addOffset(offset));
    }

    // primitive collections generator
    genList(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>] {
        const { heap, env } = this;
        const [list, listAddr, heap2] = SVObject.create(heap, source);
        const listMro = (fetchAddr(heap.getVal(env.getId('list')!)!, heap) as SVObject).getAttr('__mro__')!;
        let listVal = list;
        values.forEach((v, i) => {
            listVal = listVal.setIndice(i, v);
        });
        listVal = listVal.setAttr('$length', SVInt.create(values.length, source)).setAttr('__mro__', listMro);

        return [list, listAddr, this.setHeap(heap2.setVal(listAddr, listVal))];
    }

    genTuple(values: ShValue[], source: CodeSource | undefined): [SVObject, SVAddr, Context<T>] {
        const { heap, env } = this;
        const [tuple, tupleAddr, heap2] = SVObject.create(heap, source);
        const tupleMro = (fetchAddr(heap.getVal(env.getId('tuple')!)!, heap) as SVObject).getAttr('__mro__')!;
        let listVal = tuple;
        values.forEach((v, i) => {
            listVal = listVal.setIndice(i, v);
        });
        listVal = listVal.setAttr('$length', SVInt.create(values.length, source)).setAttr('__mro__', tupleMro);

        return [tuple, tupleAddr, this.setHeap(heap2.setVal(tupleAddr, listVal))];
    }

    // replace builtin.py source by call stack
    private _replaceBuiltinSource(source: CodeSource | undefined): CodeSource | undefined {
        if (source && !('fileId' in source)) {
            let moduleNode = source;
            while (moduleNode.nodeType !== ParseNodeType.Module) {
                moduleNode = moduleNode.parent!;
            }

            const fileInfo = getFileInfo(moduleNode)!;
            if (checkIgnorePath(fileInfo.filePath)) {
                for (let i = this.callStack.count() - 1; i >= 0; i--) {
                    const node = this.callStack.get(i)![1];
                    if (node && !('fileId' in node)) {
                        moduleNode = node;
                        while (moduleNode.nodeType !== ParseNodeType.Module) {
                            moduleNode = moduleNode.parent!;
                        }
                        const fileInfo = getFileInfo(moduleNode)!;
                        if (!checkIgnorePath(fileInfo.filePath)) {
                            return node;
                        }
                    }
                }
            }
        }

        return source;
    }
}
