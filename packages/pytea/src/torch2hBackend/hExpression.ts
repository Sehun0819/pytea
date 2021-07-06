/*
 * torchStatements.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Expressions and statements for PyTea internal languages.
 */

import { ExpressionNode } from 'pyright-internal/parser/parseNodes';

import { CodeSource } from './sharpValues';

export type HExpr = HNat | HVar | HVarTensor | HConstTensor | HLet | HTop;

export const enum HEType {
    Nat,
    Var,
    VarTensor,
    ConstTensor,
    Let,
    Top,
}

export const enum HVarTensorType {
    Plain,
    Cipher,
}

export const enum HETopType {
    Bop,
    Matmul,
    Concat,
    Reshape,
    Max,
    Maxr,
    Argmaxr,
    Sumr,
    Convol,
    Convolp,
    Maxpool,
    Maxpoolp,
    Averagepool,
    Averagepoolp,
    Map,
}

// element-wise Tensor bop
export const enum HEBopType {
    Add,
    Sub,
    Mul,
    Max,
}

// element-wise Tensor uop
export const enum HEUopType {
    Relu,
    Sigmoid,
    Rtangent,
}

interface HExprBase {
    etype: HEType;
    source?: CodeSource;
}

export interface HNat extends HExprBase {
    etype: HEType.Nat;
    value: number;
}

export namespace HNat {
    export function create(value: number, source?: CodeSource): HNat {
        return {
            etype: HEType.Nat,
            value,
            source,
        };
    }
}

export interface HVar extends HExprBase {
    etype: HEType.Var;
    ident: string;
}

export namespace HVar {
    export function create(ident: string, source?: ExpressionNode): HVar {
        return {
            etype: HEType.Var,
            ident,
            source,
        };
    }
}

export interface HVarTensor extends HExprBase {
    etype: HEType.VarTensor;
    ttype: HVarTensorType;
    tname: string;
    shape: number[];
}

export namespace HVarTensor {
    export function createPlain(tname: string, shape: number[], source?: ExpressionNode): HVarTensor {
        return {
            etype: HEType.VarTensor,
            ttype: HVarTensorType.Plain,
            tname,
            shape,
            source,
        };
    }

    export function createCipher(tname: string, shape: number[], source?: ExpressionNode): HVarTensor {
        return {
            etype: HEType.VarTensor,
            ttype: HVarTensorType.Cipher,
            tname,
            shape,
            source,
        };
    }
}

// Not implemeted yet
export interface HConstTensor extends HExprBase {
    etype: HEType.ConstTensor;
    tensor: undefined;
    shape: number[];
}

export namespace HConstTensor {
    export function create(shape: number[], source?: ExpressionNode): HConstTensor {
        return {
            etype: HEType.ConstTensor,
            tensor: undefined,
            shape,
            source,
        };
    }
}

export interface HLet extends HExprBase {
    etype: HEType.Let;
    vardecs: [string, HExpr][];
    scope: HExpr;
}

export namespace HLet {
    export function create(vardecs: [string, HExpr][], scope: HExpr, source?: CodeSource): HLet {
        return {
            etype: HEType.Let,
            vardecs,
            scope,
            source,
        };
    }
}

export interface HTop extends HExprBase {
    etype: HEType.Top;
    topType: HETopType;
}

export interface HBop extends HTop {
    etype: HEType.Top;
    topType: HETopType.Bop;
    bopType: HEBopType;
    left: HExpr;
    right: HExpr;
}

export interface HMatmul extends HTop {
    etype: HEType.Top;
    topType: HETopType.Matmul;
    left: HExpr;
    right: HExpr;
}

export interface HConcat extends HTop {
    etype: HEType.Top;
    topType: HETopType.Concat;
    left: HExpr;
    axis: number;
    right: HExpr;
}

export interface HReshape extends HTop {
    etype: HEType.Top;
    topType: HETopType.Reshape;
    base: HExpr;
    shape: number[];
}

export interface HMax extends HTop {
    etype: HEType.Top;
    topType: HETopType.Max;
    base: HExpr;
}

export interface HMaxr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxr;
    base: HExpr;
    axis: number;
}

export interface HArgmaxr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Argmaxr;
    base: HExpr;
    axis: number;
}

export interface HSumr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Sumr;
    base: HExpr;
    axis: number;
}

export interface HConvol extends HTop {
    etype: HEType.Top;
    topType: HETopType.Convol;
    base: HExpr;
    weight: HExpr;
}

export interface HConvolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Convolp;
    base: HExpr;
    weight: HExpr;
}

export interface HMaxpool extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxpool;
    base: HExpr;
    height: number;
    width: number;
}

export interface HMaxpoolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxpoolp;
    base: HExpr;
    height: number;
    width: number;
}

export interface HAveragepool extends HTop {
    etype: HEType.Top;
    topType: HETopType.Averagepool;
    base: HExpr;
    height: number;
    width: number;
}

export interface HAveragepoolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Averagepoolp;
    base: HExpr;
    height: number;
    width: number;
}

export namespace HTop {
    export function bop(bopType: HEBopType, left: HExpr, right: HExpr, source?: ExpressionNode): HBop {
        return {
            etype: HEType.Top,
            topType: HETopType.Bop,
            bopType,
            left,
            right,
            source,
        };
    }

    export function matmul(left: HExpr, right: HExpr, source?: ExpressionNode): HMatmul {
        return {
            etype: HEType.Top,
            topType: HETopType.Matmul,
            left,
            right,
            source,
        };
    }
}

export type ThLeftExpr = TEName | TEAttr | TESubscr;
export type ThExpr = TEConst | TEObject | TETuple | TECall | TELibCall | TEBinOp | TEUnaryOp | ThLeftExpr;
export type ThStmt =
    | TSPass
    | TSExpr
    | TSSeq
    | TSAssign
    | TSIf
    | TSForIn
    | TSReturn
    | TSContinue
    | TSBreak
    | TSLet
    | TSFunDef;

export const enum TEBopType {
    // numeric-op
    Add,
    Sub,
    Mul,
    Pow,
    TrueDiv,
    FloorDiv,
    Mod,
    // compare-op
    Lt,
    Lte,
    Eq,
    Neq,
    // boolean-op
    And,
    Or,
    Is,
    IsNot,
    // list-op
    In,
    NotIn,
}

export const enum TEUopType {
    Not,
    Neg,
}

export const enum TEConstType {
    Int,
    Float,
    String,
    Bool,
    None,
}

export const enum TEType {
    Const,
    Object,
    Tuple,
    Call,
    LibCall,
    BinOp,
    UnaryOp,
    Name,
    Attr,
    Subscr,
}

export const enum TSType {
    Pass,
    Expr,
    Seq,
    Assign,
    If,
    ForIn,
    Return,
    Continue,
    Break,
    Let,
    FunDef,
}

export function isNumericBop(bop: TEBopType): boolean {
    switch (bop) {
        case TEBopType.Add:
        case TEBopType.Sub:
        case TEBopType.Mul:
        case TEBopType.Pow:
        case TEBopType.TrueDiv:
        case TEBopType.FloorDiv:
        case TEBopType.Mod:
        case TEBopType.Lt:
        case TEBopType.Lte:
            return true;
        default:
            return false;
    }
}

interface ThExprBase {
    etype: TEType;
    source?: CodeSource;
}

interface ThStmtBase {
    stype: TSType;
    source?: CodeSource;
}

const _indentMult = 2;

export namespace HExpr {
    export function toString(hexpr: HExpr): string {
        return 'TODO: HExpr.toString';
    }
}

export namespace ThStmt {
    export function toString(th: ThStmt | ThExpr, indent?: number): string {
        if (`etype` in th) {
            switch (th.etype) {
                case TEType.Const:
                    return TEConst.toString(th);
                case TEType.Object:
                    return TEObject.toString(th);
                case TEType.Tuple:
                    return TETuple.toString(th);
                case TEType.Call:
                    return TECall.toString(th);
                case TEType.LibCall:
                    return TELibCall.toString(th);
                case TEType.BinOp:
                    return TEBinOp.toString(th);
                case TEType.UnaryOp:
                    return TEUnaryOp.toString(th);
                case TEType.Name:
                    return TEName.toString(th);
                case TEType.Attr:
                    return TEAttr.toString(th);
                case TEType.Subscr:
                    return TESubscr.toString(th);
            }
        } else {
            switch (th.stype) {
                case TSType.Pass:
                    return TSPass.toString(th, indent);
                case TSType.Expr:
                    return TSExpr.toString(th, indent);
                case TSType.Seq:
                    return TSSeq.toString(th, indent);
                case TSType.Assign:
                    return TSAssign.toString(th, indent);
                case TSType.If:
                    return TSIf.toString(th, indent);
                case TSType.ForIn:
                    return TSForIn.toString(th, indent);
                case TSType.Return:
                    return TSReturn.toString(th, indent);
                case TSType.Continue:
                    return TSContinue.toString(th, indent);
                case TSType.Break:
                    return TSBreak.toString(th, indent);
                case TSType.Let:
                    return TSLet.toString(th, indent);
                case TSType.FunDef:
                    return TSFunDef.toString(th, indent);
            }
        }
    }
}

// expressions
export interface TEConst extends ThExprBase {
    etype: TEType.Const;
    constType: TEConstType;
    value: number | string | boolean | undefined;
}

export namespace TEConst {
    export function create(
        constType: TEConstType,
        value: number | string | boolean | undefined,
        source?: CodeSource
    ): TEConst {
        return {
            etype: TEType.Const,
            constType,
            value,
            source,
        };
    }

    export function genStr(value: string, source?: CodeSource): TEConst {
        return TEConst.create(TEConstType.String, value, source);
    }

    export function genInt(value: number, source?: CodeSource): TEConst {
        return TEConst.create(TEConstType.Int, value, source);
    }

    export function genFloat(value: number, source?: CodeSource): TEConst {
        return TEConst.create(TEConstType.Float, value, source);
    }

    export function genBool(value: boolean, source?: CodeSource): TEConst {
        return TEConst.create(TEConstType.Bool, value, source);
    }

    export function genNone(source?: CodeSource): TEConst {
        return TEConst.create(TEConstType.None, undefined, source);
    }

    export function toString(expr: TEConst): string {
        if (expr.value === undefined) {
            return 'None';
        } else if (typeof expr.value === 'string') {
            return `"${expr.value}"`;
        }
        return expr.value.toString();
    }
}

export interface TEObject extends ThExprBase {
    etype: TEType.Object;
}

export namespace TEObject {
    export function create(source?: ExpressionNode): TEObject {
        return {
            etype: TEType.Object,
            source,
        };
    }

    export function toString(expr: TEObject): string {
        return 'OBJECT';
    }
}

export interface TETuple extends ThExprBase {
    etype: TEType.Tuple;
    values: ThExpr[];
}

export namespace TETuple {
    export function create(values: ThExpr[], source?: ExpressionNode): TETuple {
        return {
            etype: TEType.Tuple,
            values,
            source,
        };
    }

    export function toString(expr: TETuple): string {
        return `(${expr.values.map(ThStmt.toString).join(', ')})`;
    }
}

export interface TEName extends ThExprBase {
    etype: TEType.Name;
    ident: string;
}

export namespace TEName {
    export function create(ident: string, source?: ExpressionNode): TEName {
        return {
            etype: TEType.Name,
            ident,
            source,
        };
    }

    export function toString(expr: TEName): string {
        return expr.ident;
    }
}
export interface TEAttr extends ThExprBase {
    etype: TEType.Attr;
    left: ThExpr;
    right: string;
}
export namespace TEAttr {
    export function create(left: ThExpr, right: string, source?: ExpressionNode): TEAttr {
        return {
            etype: TEType.Attr,
            left,
            right,
            source,
        };
    }

    export function toString(expr: TEAttr): string {
        return `${ThStmt.toString(expr.left)}.${expr.right}`;
    }
}

export interface TESubscr extends ThExprBase {
    etype: TEType.Subscr;
    left: ThExpr;
    right: ThExpr;
}
export namespace TESubscr {
    export function create(left: ThExpr, right: ThExpr, source?: ExpressionNode): TESubscr {
        return {
            etype: TEType.Subscr,
            left,
            right,
            source,
        };
    }

    export function toString(expr: TESubscr): string {
        return `${ThStmt.toString(expr.left)}[${ThStmt.toString(expr.right)}]`;
    }
}
export interface TECall extends ThExprBase {
    etype: TEType.Call;
    func: ThExpr;
    params: ThExpr[];
}
export namespace TECall {
    export function create(func: ThExpr, params: ThExpr[], source?: ExpressionNode): TECall {
        return {
            etype: TEType.Call,
            func,
            params,
            source,
        };
    }

    export function toString(expr: TECall): string {
        return `${ThStmt.toString(expr.func)}(${expr.params.map(ThStmt.toString).join(', ')})`;
    }
}

export enum LibCallType {
    DEBUG = 'DEBUG',
    import = 'import',
    genList = 'genList',
    genDict = 'genDict',
    setDefault = 'setDefault',
    callKV = 'callKV',
    objectClass = 'objectClass',
    exportGlobal = 'exportGlobal',
    raise = 'raise',
    explicit = 'explicit',
}
export interface TELibCall extends ThExprBase {
    etype: TEType.LibCall;
    type: LibCallType;
    params: [string, ThExpr][];
}
export namespace TELibCall {
    export function create(type: LibCallType, params: [string, ThExpr][], source?: CodeSource): TELibCall {
        return {
            etype: TEType.LibCall,
            type,
            params,
            source,
        };
    }

    export function toString(expr: TELibCall): string {
        return `LIBCALL(${expr.type}, ${expr.params
            .map(([pn, expr]) => (pn ? `${pn}=${ThStmt.toString(expr)}` : ThStmt.toString(expr)))
            .join(', ')})`;
    }
}

export interface TEBinOp extends ThExprBase {
    etype: TEType.BinOp;
    bopType: TEBopType;
    left: ThExpr;
    right: ThExpr;
}
export namespace TEBinOp {
    export function create(bopType: TEBopType, left: ThExpr, right: ThExpr, source?: ExpressionNode): TEBinOp {
        return {
            etype: TEType.BinOp,
            bopType,
            left,
            right,
            source,
        };
    }

    export function toString(expr: TEBinOp): string {
        return `(${ThStmt.toString(expr.left)} ${TEBinOp.toStringBop(expr.bopType)} ${ThStmt.toString(expr.right)})`;
    }

    export function toStringBop(bop: TEBopType): string {
        switch (bop) {
            case TEBopType.Add:
                return '+';
            case TEBopType.Sub:
                return '-';
            case TEBopType.Mul:
                return '*';
            case TEBopType.Pow:
                return '**';
            case TEBopType.TrueDiv:
                return '/';
            case TEBopType.FloorDiv:
                return '//';
            case TEBopType.Mod:
                return '%';
            case TEBopType.Lt:
                return '<';
            case TEBopType.Lte:
                return '<=';
            case TEBopType.Eq:
                return '==';
            case TEBopType.Neq:
                return '!=';
            case TEBopType.And:
                return 'and';
            case TEBopType.Or:
                return 'or';
            case TEBopType.Is:
                return 'is';
            case TEBopType.IsNot:
                return 'is not';
            case TEBopType.In:
                return 'in';
            case TEBopType.NotIn:
                return 'not in';
        }
    }
}
export interface TEUnaryOp extends ThExprBase {
    etype: TEType.UnaryOp;
    uopType: TEUopType;
    base: ThExpr;
}
export namespace TEUnaryOp {
    export function create(uopType: TEUopType, base: ThExpr, source?: ExpressionNode): TEUnaryOp {
        return {
            etype: TEType.UnaryOp,
            uopType,
            base,
            source,
        };
    }

    export function toString(expr: TEUnaryOp): string {
        const unary = expr.uopType === TEUopType.Neg ? '-' : 'not ';
        return `(${unary}${ThStmt.toString(expr.base)})`;
    }
}

// statements
function sp(indent?: number): string {
    return indent ? ' '.repeat(indent) : '';
}

export interface TSPass extends ThStmtBase {
    stype: TSType.Pass;
}
export namespace TSPass {
    const _pass: TSPass = { stype: TSType.Pass };
    export function get(source?: CodeSource): TSPass {
        if (!source) {
            return _pass;
        }
        return {
            stype: TSType.Pass,
            source,
        };
    }

    export function toString(stmt: TSPass, indent?: number): string {
        return `${sp(indent)}pass`;
    }
}

export interface TSExpr extends ThStmtBase {
    stype: TSType.Expr;
    expr: ThExpr;
}
export namespace TSExpr {
    export function create(expr: ThExpr): TSExpr {
        return {
            stype: TSType.Expr,
            expr,
            source: expr.source,
        };
    }

    export function toString(stmt: TSExpr, indent?: number): string {
        return `${sp(indent)}${ThStmt.toString(stmt.expr)}`;
    }
}

export interface TSSeq extends ThStmtBase {
    stype: TSType.Seq;
    left: ThStmt;
    right: ThStmt;
}
export namespace TSSeq {
    export function create(left: ThStmt, right: ThStmt, source?: CodeSource): TSSeq {
        return {
            stype: TSType.Seq,
            left,
            right,
            source,
        };
    }

    export function toString(stmt: TSSeq, indent?: number): string {
        return `${ThStmt.toString(stmt.left, indent)};\n${ThStmt.toString(stmt.right, indent)}`;
    }
}

export interface TSAssign extends ThStmtBase {
    stype: TSType.Assign;
    left: ThLeftExpr;
    right: ThExpr;
}
export namespace TSAssign {
    export function create(left: ThLeftExpr, right: ThExpr, source?: CodeSource): TSAssign {
        return {
            stype: TSType.Assign,
            left,
            right,
            source,
        };
    }

    export function toString(stmt: TSAssign, indent?: number): string {
        return `${sp(indent)}${ThStmt.toString(stmt.left)} = ${ThStmt.toString(stmt.right)}`;
    }
}

export interface TSIf extends ThStmtBase {
    stype: TSType.If;
    cond: ThExpr;
    thenStmt: ThStmt;
    elseStmt: ThStmt;
}
export namespace TSIf {
    export function create(cond: ThExpr, thenStmt: ThStmt, elseStmt: ThStmt, source?: CodeSource): TSIf {
        return {
            stype: TSType.If,
            cond,
            thenStmt,
            elseStmt,
            source,
        };
    }

    export function toString(stmt: TSIf, indent?: number): string {
        const i = indent ? indent : 0;
        return `${sp(i)}if ${ThStmt.toString(stmt.cond)} then {${sp(i)}\n${ThStmt.toString(
            stmt.thenStmt,
            i + _indentMult
        )}\n${sp(i)}} else {\n${sp(i)}${ThStmt.toString(stmt.elseStmt, i + _indentMult)}\n${sp(i)}}`;
    }
}

export interface TSForIn extends ThStmtBase {
    stype: TSType.ForIn;
    ident: string;
    loopVal: ThExpr;
    loopBody: ThStmt;
}
export namespace TSForIn {
    export function create(ident: string, loopVal: ThExpr, loopBody: ThStmt, source?: CodeSource): TSForIn {
        return {
            stype: TSType.ForIn,
            ident,
            loopVal,
            loopBody,
            source,
        };
    }

    export function toString(stmt: TSForIn, indent?: number): string {
        const i = indent ? indent : 0;
        return `${sp(i)}for ${stmt.ident} in ${ThStmt.toString(stmt.loopVal)} {\n${sp(i)}${ThStmt.toString(
            stmt.loopBody,
            i + _indentMult
        )}\n${sp(i)}}`;
    }
}

export interface TSReturn extends ThStmtBase {
    stype: TSType.Return;
    expr: ThExpr;
}
export namespace TSReturn {
    export function create(expr: ThExpr, source?: CodeSource): TSReturn {
        return {
            stype: TSType.Return,
            expr,
            source,
        };
    }

    export function toString(stmt: TSReturn, indent?: number): string {
        return `${sp(indent)}return ${ThStmt.toString(stmt.expr)}`;
    }
}

export interface TSContinue extends ThStmtBase {
    stype: TSType.Continue;
}
export namespace TSContinue {
    export function create(source?: CodeSource): TSContinue {
        return {
            stype: TSType.Continue,
            source,
        };
    }

    export function toString(stmt: TSContinue, indent?: number): string {
        return `${sp(indent)}continue`;
    }
}

export interface TSBreak extends ThStmtBase {
    stype: TSType.Break;
}
export namespace TSBreak {
    export function create(source?: CodeSource): TSBreak {
        return {
            stype: TSType.Break,
            source,
        };
    }

    export function toString(stmt: TSBreak, indent?: number): string {
        return `${sp(indent)}break`;
    }
}

export interface TSLet extends ThStmtBase {
    stype: TSType.Let;
    name: string;
    expr?: ThExpr;
    scope: ThStmt;
}
export namespace TSLet {
    export function create(name: string, scope: ThStmt, expr?: ThExpr, source?: CodeSource): TSLet {
        return {
            stype: TSType.Let,
            name,
            expr,
            scope,
            source,
        };
    }

    export function toString(stmt: TSLet, indent?: number): string {
        const i = indent ? indent : 0;
        return `${sp(i)}${stmt.name} := ${stmt.expr ? ThStmt.toString(stmt.expr) : 'undef'} in \n${ThStmt.toString(
            stmt.scope,
            indent
        )}`;
    }
}

export interface TSFunDef extends ThStmtBase {
    stype: TSType.FunDef;
    name: string;
    params: string[];
    body: ThStmt;
    scope: ThStmt;
    hasClosure: boolean;
}
export namespace TSFunDef {
    export function create(name: string, params: string[], body: ThStmt, scope: ThStmt, source?: CodeSource): TSFunDef {
        return {
            stype: TSType.FunDef,
            name,
            params,
            body,
            scope,
            source,
            hasClosure: findClosure(body),
        };
    }

    export function toString(stmt: TSFunDef, indent?: number): string {
        const i = indent ? indent : 0;
        return `${sp(i)}def ${stmt.name}(${stmt.params.join(', ')}) {\n${ThStmt.toString(
            stmt.body,
            i + _indentMult
        )}\n${sp(i)}}\n${ThStmt.toString(stmt.scope, indent)}`;
    }

    export function findClosure(stmt: ThStmt): boolean {
        switch (stmt.stype) {
            case TSType.FunDef:
                return true;
            case TSType.Seq:
                return findClosure(stmt.left) || findClosure(stmt.right);
            case TSType.Let:
                return findClosure(stmt.scope);
            case TSType.If:
                return findClosure(stmt.thenStmt) || findClosure(stmt.elseStmt);
            case TSType.ForIn:
                return findClosure(stmt.loopBody);
            default:
                return false;
        }
    }
}
