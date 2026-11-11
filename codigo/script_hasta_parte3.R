# INSTALACIÓN DE PAQUETES Y LECTURA DE DATASET:

# Paquetes que necesitamos para todo el proyecto:
paquetes <- c(
  "tidyverse",   # manipulación de datos y gráficos
  "caret",       # partición, validación cruzada y modelos
  "janitor",     # limpieza de nombres y tablas
  "rpart",       # árboles de clasificación
  "rpart.plot",  # visualización de árboles
  "knitr",       # tablas en Markdown
  "dplyr",
  "Metrics"      # Para calcular RMSE, MAE fácilmente
)

# Instalamos solo los que falten (se ejecuta una sola vez por máquina)
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) {
  install.packages(nuevos)
}

# Cargamos todas las librerías. invisible() evita que se imprima en pantalla
# la lista de TRUE/FALSE que devuelve lapply() al cargar cada una.
invisible(lapply(paquetes, library, character.only = TRUE))

# Se establece la ruta del dataset:
ruta <- "../datos/used_cars.csv"

if (!file.exists(ruta)) {
  stop(
    "No se encontró ", ruta, ". "
  )
}

# Se leen los datos:
datos <- read.csv(ruta, header = TRUE)



# ==============================================================================================================================
# ==============================================================================================================================



# PARTICIÓN TRAIN / TEST (se hace ANTES de cualquier limpieza o transformación):

# Carga y partición de datos:

# Suponiendo que el dataset ya está cargado como 'datos'
# y que la variable respuesta es 'price'

# El split se hace sobre los datos crudos, de modo que ninguna decisión de limpieza
# se calcule mirando el set de test. A partir de aquí, 'test' no se toca
# hasta la evaluación final.

set.seed(123)  # Reproducibilidad

n <- nrow(datos)
indices_train <- sample(1:n, size = 0.8 * n)

train <- datos[indices_train, ]
test  <- datos[-indices_train, ]



# ==============================================================================================================================
# ==============================================================================================================================



# LIMPIEZA DE DATOS:

# Las reglas de limpieza son FIJAS: no se aprenden de los datos (convertir símbolos en NA,
# reemplazar NA por la etiqueta "unknown", quitar "$" y "mi."). Se definen una sola vez
# en una función y se aplican por separado a train y a test.
# La exploración (conteos, rangos, resúmenes) se hace SOLO sobre train.

# Se revisan los primeros detos del conjunto:
head(train)
str(train)
dim(train)

# Valores faltantes:
colSums(is.na(train))    # No hay valores NA en este dataset

# Valores "unknown" por columna:
sapply(train, function(x) sum(x == "unknown")) # No hay valores "unknown"

# Contar cadenas vacías por columna
sapply(train, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "", na.rm = TRUE)
  } else {
    0
  }
})

# Encontrar el tipo de "guion" presente en las casillas, ya que no es el típico:
# (se busca en train la primera casilla de ext_col con un solo carácter)
pos_guion <- which(nchar(train$ext_col) == 1)[1]
train$ext_col[pos_guion]
charToRaw(train$ext_col[pos_guion])

# Contar guiones por columna
sapply(train, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "–", na.rm = TRUE)
  } else {
    0
  }
})

# Copia temporal de train, solo para mirar cuántos NA aparecen al convertir
# las cadenas vacías y los guiones (no se usa después):
train_na <- train
train_na[train_na == ""] <- NA
train_na[train_na == "–"] <- NA

# Cantidad de NA por columna
colSums(is.na(train_na))
# Porcentaje de NA por columna
round(colMeans(is.na(train_na)) * 100, 2)
rm(train_na)

# Reglas de limpieza (idénticas para train y test):
limpiar_datos <- function(df) {
  
  # Cadenas vacías y guiones pasan a NA
  df[df == ""] <- NA
  df[df == "–"] <- NA
  
  # Se intercambian los NA de la variable NA por "unknown" ya que si se eliminaran las filas con datos nulos,
  # se perdería una cantidad considerable de información:
  df$clean_title[is.na(df$clean_title)] <- "unknown"
  df$accident[is.na(df$accident)] <- "unknown"
  df$fuel_type[is.na(df$fuel_type)] <- "unknown"
  df$engine[is.na(df$engine)] <- "unknown"
  df$transmission[is.na(df$transmission)] <- "unknown"
  df$ext_col[is.na(df$ext_col)] <- "unknown"
  df$int_col[is.na(df$int_col)] <- "unknown"
  
  # Traspasar datos numéricos a clase numeric:
  
  # Para price y milage, hay que limpiar la variable para que no imprima error por los símbolos:
  df$price <- as.numeric(
    gsub("[$,]", "", trimws(as.character(df$price)))
  )
  
  df$milage <- as.numeric(
    gsub(",", "",
         gsub(" mi\\.", "", as.character(df$milage)))
  )
  
  # En el caso del año del modelo, se hace conversión directa:
  df$model_year <- as.numeric(as.character(df$model_year))
  
  df
}

train <- limpiar_datos(train)
test  <- limpiar_datos(test)

# Valores "unknown" por columna:
sapply(train, function(x) sum(x == "unknown"))

# Duplicados:
sum(duplicated(train)) # No hay duplicados

# Control de integridad (no modifica nada): filas idénticas dentro de train, dentro de test
# o repetidas entre ambos. Un 0 confirma que ninguna fila de test "está escondida" en train.
sum(duplicated(rbind(train, test)))

# Rangos razonables:
range(train$model_year) # 1974 hasta 2024
range(train$milage)  # 1,000 mi hasta 99,999 mi
range(train$price)   # $1,599,000 hasta $99,999


# No se eliminan variables, todas sirven


summary(train)
names(train)



# ==============================================================================================================================
# ==============================================================================================================================



# BASELINES:

# Baseline 1: Media global

media_global <- mean(train$price, na.rm = TRUE)

pred_media <- rep(media_global, nrow(test))

rmse_media <- rmse(test$price, pred_media)
mae_media  <- mae(test$price, pred_media)

cat("Baseline - Media global\n")
cat("RMSE:", rmse_media, "\n")
cat("MAE :", mae_media, "\n\n")

# Baseline 2: Valor previo (persistencia)

# Dado que no existe un orden temporal real en el dataset,
# este baseline se construye utilizando el orden original de las
# observaciones como aproximación. Su interpretación debe tomarse
# con cautela y mencionarse como limitación en el informe.

# Para no usar precios de test como entrada, cada fila de test se predice con el precio
# de la fila de TRAIN más reciente que la precede en el orden original
# (los precios de las otras filas de test son desconocidos al momento de predecir).
id_train <- as.integer(rownames(train))   # posición original de cada fila de train
id_test  <- as.integer(rownames(test))    # posición original de cada fila de test

train_ordenado <- train[order(id_train), ]
id_train_ord   <- sort(id_train)

# findInterval(): para cada fila de test, cuenta cuántas filas de train la preceden
pos_previa <- findInterval(id_test, id_train_ord)

# Si no hay ninguna fila de train antes (test en la primera posición), se usa la media global
pred_vp <- ifelse(pos_previa == 0, media_global,
                  train_ordenado$price[pmax(pos_previa, 1)])
real_vp <- test$price

rmse_vp <- rmse(real_vp, pred_vp)
mae_vp  <- mae(real_vp, pred_vp)

cat("Baseline - Valor previo (persistencia)\n")
cat("RMSE:", rmse_vp, "\n")
cat("MAE :", mae_vp, "\n\n")



#Baseline 3: Promedio por marca:
promedio_por_marca <- train %>%
  group_by(brand) %>%
  summarise(
    precio_promedio = mean(price, na.rm = TRUE),
    .groups = "drop"
  )

# Unir el promedio de cada marca al conjunto de prueba
test_grupo <- test %>%
  left_join(promedio_por_marca, by = "brand")

# Para marcas no vistas en entrenamiento, usar la media global
test_grupo$precio_promedio[
  is.na(test_grupo$precio_promedio)
] <- media_global

# Calcular métricas
rmse_grupo <- rmse(
  test_grupo$price,
  test_grupo$precio_promedio
)

mae_grupo <- mae(
  test_grupo$price,
  test_grupo$precio_promedio
)

cat("Baseline - Promedio por grupo (marca)\n")
cat("RMSE:", rmse_grupo, "\n")
cat("MAE :", mae_grupo, "\n\n")


#Resumen comparativo de los baselines:
resumen_baselines <- data.frame(
  Modelo = c("Media global", "Valor previo", "Promedio por grupo (marca)"),
  RMSE   = c(rmse_media, rmse_vp, rmse_grupo),
  MAE    = c(mae_media, mae_vp, mae_grupo)
)

print(resumen_baselines) # Comparar cuál baseline tiene menos error



# ==============================================================================================================================
# ==============================================================================================================================



# ==============================================================================
# ROL 3: ENSAMBLES (Árbol de Regresión, Random Forest, XGBoost y Modelo Combinado)
# ==============================================================================
#
# IDEA GENERAL (para la presentación oral):
# Un "modelo de ensamble" es como pedirle el precio de un auto a VARIOS expertos
# en lugar de a uno solo. Cada experto (modelo) se equivoca un poco distinto, y
# al juntar sus opiniones el error total suele bajar.
#
# Entrenaremos 3 expertos individuales:
#   1) Árbol de regresión : un solo árbol de decisiones (simple y fácil de explicar).
#   2) Random Forest      : cientos de árboles que "votan" (promedian) el precio.
#   3) XGBoost            : árboles en fila, donde cada uno corrige los errores del anterior.
# Y luego un 4º modelo que COMBINA a los tres anteriores.
#
# Todo se hace con el paquete "caret", que es como un "control remoto universal":
# con la misma función train() podemos entrenar modelos muy distintos.
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. PAQUETES ADICIONALES
# ------------------------------------------------------------------------------

# caretEnsemble : permite combinar varios modelos de caret en uno solo.
# ranger        : motor rápido que usa caret para el Random Forest.
# xgboost       : motor que usa caret para XGBoost.
extra <- c("caretEnsemble", "ranger", "xgboost")

# Revisamos cuáles de esos paquetes NO están instalados en este computador...
faltan <- extra[!(extra %in% installed.packages()[, "Package"])]

# ...y si falta alguno, lo instalamos (si no falta ninguno, no hace nada).
if (length(faltan) > 0) install.packages(faltan)

# Cargamos los tres paquetes (invisible() solo evita que se imprima basura en pantalla).
invisible(lapply(extra, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 1. PREPARACIÓN DE LOS DATOS PARA LOS MODELOS
# ------------------------------------------------------------------------------

# Ojo: en este script tenemos un data frame llamado "train" (los datos de
# entrenamiento) y caret tiene una función llamada train(). No se pisan:
# R distingue si usamos "train" como dato o train(...) como función.

# --- 1.1 Variables que NO usaremos --------------------------------------------
# Los modelos no entienden texto. Para usar una variable de texto (por ejemplo
# la marca), R la convierte en columnas de 0 y 1, una por cada categoría.
# Eso se llama "dummy encoding" (codificación dummy). Ejemplo:
#     marca = "BMW"  ->  columna_BMW = 1, columna_Audi = 0, columna_Ford = 0, ...
# El problema: si una variable tiene cientos de categorías distintas (como
# "model", "ext_col" o "int_col": modelo del auto y colores), se crean cientos
# de columnas nuevas y el computador se queda sin memoria. Por eso las excluimos.
vars_excluir <- c("model", "ext_col", "int_col")

# Nos quedamos con todas las columnas MENOS las excluidas (el signo ! significa "no").
train_m <- train[, !(names(train) %in% vars_excluir)]
test_m  <- test[,  !(names(test)  %in% vars_excluir)]

# --- 1.2 Convertir texto a "factor" -------------------------------------------
# Un "factor" es la forma en que R guarda variables de categorías.
# Buscamos todas las columnas que son texto (character)...
cols_chr <- names(train_m)[sapply(train_m, is.character)]

# ...y para cada una (for = "repite esto por cada columna") hacemos lo siguiente:
for (v in cols_chr) {
  # Lista de las categorías que aparecen en train, sin repetir, más "unknown"
  # (que usaremos como comodín). Ya NO miramos las categorías de test al armar
  # esta lista: así el set de test sigue intocable hasta la evaluación final.
  niveles <- sort(unique(c(train_m[[v]], "unknown")))
  
  # Si en test aparece una categoría que train nunca vio (por ejemplo, un motor
  # raro), el modelo no tiene una columna dummy para ella y se caería con un error
  # de "categoría nueva". En ese caso la tratamos como "unknown": es decir,
  # "una categoría que el modelo no conoce".
  test_m[[v]][!(test_m[[v]] %in% niveles)] <- "unknown"
  
  # NUEVO: "limpiamos" los nombres de las categorías. make.names() reemplaza
  # espacios, guiones, barras y otros símbolos raros por puntos
  # (ej.: "6-Speed A/T" pasa a ser "X6.Speed.A.T"; los nombres que empiezan con
  # un número reciben una "X" delante). Así los nombres de las columnas dummy
  # quedan con caracteres "seguros" para los motores de los modelos.
  # unique = TRUE evita que dos categorías distintas terminen con el mismo
  # nombre limpio (si ocurre, agrega .1, .2, etc.).
  # Como usamos la MISMA lista (niveles) para train y test, la traducción de
  # nombres es idéntica en ambos y las categorías siguen coincidiendo.
  niveles_limpios <- make.names(niveles, unique = TRUE)
  
  # Convertimos la columna a factor usando esa misma lista en train y en test.
  # levels = nombres originales (para reconocer cada categoría);
  # labels = nombres limpios (los que verá el modelo).
  train_m[[v]] <- factor(train_m[[v]], levels = niveles, labels = niveles_limpios)
  test_m[[v]]  <- factor(test_m[[v]],  levels = niveles, labels = niveles_limpios)
}


# ------------------------------------------------------------------------------
# 2. VALIDACIÓN CRUZADA (cómo elegimos la mejor configuración de cada modelo)
# ------------------------------------------------------------------------------

# ¿QUÉ ES LA VALIDACIÓN CRUZADA DE 5 FOLDS?
# Imagina que los datos de entrenamiento son un mazo de cartas.
#   1. Lo dividimos al azar en 5 montones iguales (los "folds").
#   2. Entrenamos el modelo con 4 montones y lo "examinamos" con el 5º, que no vio.
#   3. Repetimos 5 veces, rotando cuál montón hace de examen.
#   4. Promediamos las 5 notas. Esa nota promedio dice qué tan bien generaliza
#      el modelo a datos nuevos.
# Así probamos muchas configuraciones (hiperparámetros) y elegimos la mejor,
# SIN tocar el set de test, que se reserva para la evaluación final.

# Creamos los 5 montones UNA sola vez, con semilla fija (set.seed(123) hace que
# el "azar" sea siempre el mismo y los resultados se puedan repetir).
# Usar los mismos montones en todos los modelos es obligatorio para poder
# combinarlos después con caretEnsemble. returnTrain = TRUE devuelve, para cada
# fold, las filas que se usan para ENTRENAR (los otros 4 montones).
# Todos los modelos del proyecto (incluidos los del resto del equipo) deben usar
# este mismo objeto "folds" y la misma hoja "ctrl", para que la comparación
# final de la tabla sea justa.
set.seed(123)
folds <- createFolds(train_m$price, k = 5, returnTrain = TRUE)

# trainControl() es la "hoja de instrucciones" que se le pasa a train():
ctrl <- trainControl(
  method          = "cv",        # "cv" = validación cruzada
  number          = 5,           # 5 folds (5 montones)
  index           = folds,       # usar exactamente los montones que creamos arriba
  savePredictions = "final"      # guardar las predicciones de "examen" del mejor modelo;
  # sin esto, caretEnsemble no puede combinar los modelos
)

# --- Preprocesamiento SOLO dentro de train() ----------------------------------
# DATA LEAKAGE (fuga de información): es "hacer trampa sin querer", por ejemplo
# calcular el promedio de una variable usando también los datos de examen.
# Para evitarlo, el preprocesamiento se declara en el argumento preProcess de
# train(), y caret lo calcula dentro de cada fold, usando solo sus datos de
# entrenamiento. Qué hace cada opción:
#   "zv"     : (zero variance) elimina columnas que valen siempre lo mismo; no sirven.
#   "center" : resta el promedio a cada variable (queda centrada en 0).
#   "scale"  : divide por la desviación estándar (todas las variables quedan en la misma escala).
pre <- c("zv", "center", "scale")


# ------------------------------------------------------------------------------
# 3. DIAGNÓSTICO DE DIMENSIONALIDAD (cuántas columnas ve realmente el modelo)
# ------------------------------------------------------------------------------

# Después de quitar price (la variable a predecir) y las 3 variables excluidas,
# quedan 8 predictores "base". PERO el modelo no ve 8 columnas: ve las columnas
# que resultan del dummy encoding. Una variable con 50 categorías aporta 49
# columnas. Variables como brand, engine y transmission tienen muchas categorías,
# así que el total real es mucho mayor que 8.
# Este dato importa porque Random Forest tiene un parámetro (mtry) que NO puede
# ser mayor que el número real de columnas. Por eso lo calculamos en vez de adivinarlo.

# model.matrix() hace el dummy encoding igual que lo haría caret por dentro.
# El [, -1] elimina la primera columna (el "intercepto", que no es un predictor).
x_dummy <- model.matrix(price ~ ., data = train_m)[, -1, drop = FALSE]

# Número total de columnas después del dummy encoding.
p_total <- ncol(x_dummy)

# Dentro de cada fold, la opción "zv" elimina las columnas que quedan constantes
# (por ejemplo, una categoría que no aparece en las filas de ese fold).
# Entonces, el número de columnas útiles puede ser un poco menor en cada fold.
# Contamos cuántas columnas NO constantes tiene cada fold y nos quedamos con el
# MÍNIMO: ese es el límite seguro para mtry.
p_folds <- vapply(
  folds,
  function(idx) {
    # Para cada columna vemos si tiene al menos un valor distinto del primero
    # (si lo tiene, no es constante). sum() cuenta cuántas columnas cumplen.
    sum(apply(x_dummy[idx, , drop = FALSE], 2, function(col) any(col != col[1])))
  },
  numeric(1)
)
p_min <- min(p_folds)

# Mostramos los números en pantalla para poder citarlos en la presentación.
cat("Predictores base:", ncol(train_m) - 1,
    "| Columnas tras dummy encoding:", p_total,
    "| Mínimo de columnas útiles por fold (tras zv):", p_min, "\n")


# ------------------------------------------------------------------------------
# 4. FUNCIÓN PARA MEDIR QUÉ TAN BIEN PREDICE CADA MODELO EN TEST
# ------------------------------------------------------------------------------

# Una "función" es una receta reutilizable: le damos las predicciones y los
# precios reales, y nos devuelve las 3 métricas. Así no repetimos código 5 veces.
#
# Las métricas:
#   RMSE : error típico en dólares. Eleva los errores al cuadrado, así que
#          castiga MUCHO los errores grandes (equivocarse por $50.000 pesa mucho).
#   MAE  : error promedio absoluto en dólares ("en promedio me equivoco por $X").
#          Es más fácil de explicar que el RMSE.
#   R²   : porcentaje de la variación del precio que el modelo logra explicar.
#          1 = perfecto, 0 = tan malo como predecir siempre el promedio
#          (puede ser negativo si es peor que eso).
evaluar <- function(pred, obs, nombre) {
  # Aseguramos que las predicciones sean un vector de números simple
  # (los modelos combinados a veces las entregan en otro formato).
  pred <- as.numeric(unlist(pred))
  
  data.frame(
    Modelo = nombre,
    RMSE   = Metrics::rmse(obs, pred),
    MAE    = Metrics::mae(obs, pred),
    R2     = caret::R2(pred, obs, form = "traditional")  # R² clásico: 1 - (error del modelo / error de predecir el promedio)
  )
}


# ------------------------------------------------------------------------------
# 5. MODELO 1: ÁRBOL DE REGRESIÓN (rpart)
# ------------------------------------------------------------------------------

# Un árbol de regresión es una cadena de preguntas de sí/no, por ejemplo:
# "¿el año es mayor a 2018?" -> "¿el kilometraje es menor a 30.000?" -> ...
# Al final de cada rama predice el precio promedio de los autos que llegaron ahí.

# Grilla = lista de valores de hiperparámetros que caret va a probar.
# Un hiperparámetro es una "perilla" del modelo que nosotros fijamos antes de entrenar.
grid_rpart <- expand.grid(
  # cp (parámetro de complejidad): es el "castigo por crecer". Una pregunta nueva
  # en el árbol solo se acepta si mejora el ajuste al menos en este porcentaje.
  #   cp pequeño (0.0001) -> árbol enorme y detallado (riesgo de memorizar los datos).
  #   cp grande  (0.02)   -> árbol chico y simple (riesgo de quedar muy simplista).
  # En rpart la "profundidad" del árbol se controla de forma indirecta con este valor.
  cp = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02)
)

set.seed(123)   # misma "suerte" cada vez que se ejecute
mod_rpart <- train(
  price ~ .,                # fórmula: "predecir price usando todas las demás columnas"
  data       = train_m,     # datos con los que se entrena
  method     = "rpart",     # tipo de modelo: árbol de regresión
  metric     = "RMSE",      # criterio para elegir el mejor cp: el menor RMSE en validación cruzada
  trControl  = ctrl,        # hoja de instrucciones de la validación cruzada (5 folds)
  tuneGrid   = grid_rpart,  # valores de cp que se van a probar
  preProcess = pre          # preprocesamiento hecho DENTRO de cada fold (sin fuga de datos)
)


# ------------------------------------------------------------------------------
# 6. MODELO 2: RANDOM FOREST (ranger)
# ------------------------------------------------------------------------------

# Random Forest = "bosque aleatorio": entrena muchos árboles, cada uno con una
# muestra distinta de filas y mirando solo algunas columnas en cada decisión.
# El precio final es el PROMEDIO de todos los árboles. Como cada árbol se
# equivoca en cosas distintas, al promediar los errores se compensan.

# --- Cálculo de mtry (usa el p_min que calculamos en la sección 3) ------------
# mtry = cuántas columnas "sortea" cada árbol para elegir CADA pregunta.
#   mtry bajo  -> los árboles se parecen menos entre sí (más variedad), pero
#                 cada uno ve poca información.
#   mtry alto  -> cada árbol ve más información, pero se parecen más entre sí.
# Reglas clásicas: raíz cuadrada de p (extremo bajo) y p/3 (valor típico en
# regresión). Agregamos p/6 como punto intermedio.
# floor() redondea hacia abajo, porque mtry debe ser un número entero.
mtry_cand <- floor(c(sqrt(p_min), p_min / 6, p_min / 3))

# pmax(..., 1) asegura que mtry sea al menos 1; pmin(..., p_min) asegura que
# nunca sea mayor que las columnas disponibles (si lo fuera, ranger da error).
# unique() elimina repetidos y sort() los ordena de menor a mayor.
mtry_cand <- sort(unique(pmin(pmax(mtry_cand, 1), p_min)))

grid_ranger <- expand.grid(
  mtry          = mtry_cand,   # columnas sorteadas en cada pregunta (explicado arriba)
  splitrule     = "variance",  # regla para decidir cada pregunta: elegir la que más reduce la varianza (dispersión) del precio
  min.node.size = c(3, 5, 10)  # mínimo de autos que debe tener un grupo final (hoja):
  #   valor chico -> árboles muy detallados (riesgo de memorizar);
  #   valor grande -> árboles más suaves (riesgo de simplificar de más)
)

# Seguro de vida: si por algún motivo la grilla tuviera un mtry inválido,
# el programa se detiene aquí, ANTES de gastar tiempo en la validación cruzada.
stopifnot(all(grid_ranger$mtry >= 1), all(grid_ranger$mtry <= p_min))

set.seed(123)
mod_ranger <- train(
  price ~ .,
  data       = train_m,
  method     = "ranger",    # motor de Random Forest
  metric     = "RMSE",
  trControl  = ctrl,
  tuneGrid   = grid_ranger,
  preProcess = pre,
  num.trees  = 300,         # número de árboles del bosque. Más árboles = resultado más estable
  # pero más lento. Es fijo (no se prueba en la grilla).
  importance = "impurity",  # pide calcular qué variables fueron más importantes para el bosque
  verbose    = FALSE        # NUEVO: silencia los mensajes "Growing trees.. Progress: X%" que ranger
  # imprime cada vez que entrena un bosque (solo afecta lo que se ve en
  # pantalla; no cambia los resultados del modelo)
)


# ------------------------------------------------------------------------------
# 7. MODELO 3: XGBOOST (método propio compatible con xgboost 3.x)
# ------------------------------------------------------------------------------

# XGBoost = "boosting": los árboles se construyen EN FILA. El árbol 1 hace una
# predicción tosca; el árbol 2 se entrena para corregir los errores del 1; el 3
# corrige los errores que quedan, y así sucesivamente. Es como corregir un
# examen una y otra vez hasta que queda casi perfecto. A diferencia del Random
# Forest (árboles independientes que votan), aquí cada árbol depende del anterior.

# --- Por qué NO usamos method = "xgbTree" -------------------------------------
# El método "xgbTree" que trae caret fue escrito para versiones antiguas de
# xgboost. Con xgboost 3.x (la que tenemos instalada) falla en cada fold: caret
# intenta "anotarle" información al modelo entrenado y xgboost 3.x ya no lo
# permite. El error que vimos fue: "ALTLIST classes must provide a Set_elt method".
# Solución: le explicamos a caret, con una "receta" propia, cómo entrenar y
# predecir con xgboost. Sigue siendo train() de caret (misma validación cruzada,
# mismo preProcess, mismos folds), solo cambia quién ejecuta XGBoost por dentro.

# Una receta de modelo en caret es una lista con las piezas que train() necesita:
xgb_custom <- list(
  label   = "XGBoost compatible con xgboost 3.x",
  library = "xgboost",          # paquete que debe estar instalado
  type    = "Regression",       # predecimos un número (precio), no una categoría
  
  # Lista de hiperparámetros que este modelo acepta (nombre, tipo y descripción).
  parameters = data.frame(
    parameter = c("nrounds", "max_depth", "eta", "gamma",
                  "colsample_bytree", "min_child_weight", "subsample"),
    class     = rep("numeric", 7),
    label     = c("Rondas de boosting", "Profundidad maxima", "Tasa de aprendizaje",
                  "Ganancia minima", "Fraccion de columnas", "Peso minimo hijo",
                  "Fraccion de filas")
  ),
  
  # Grilla por defecto (no se usa, porque nosotros entregamos tuneGrid propio,
  # pero caret exige que exista).
  grid = function(x, y, len = NULL, search = "grid") {
    expand.grid(nrounds = 100, max_depth = 3, eta = 0.1, gamma = 0,
                colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
  },
  
  loop = NULL,                  # sin atajos: se entrena una vez por cada combinación
  
  # CÓMO SE ENTRENA: caret nos pasa x (columnas ya preprocesadas), y (precios)
  # y param (una fila de la grilla con los valores a probar).
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    # xgboost necesita sus datos en un formato propio ("DMatrix"): una matriz numérica + la respuesta.
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = y)
    
    # Hiperparámetros agrupados en una lista. Aquí SÍ va el objetivo
    # (que era lo que causaba el aviso de "objective"):
    #   reg:squarederror = regresión minimizando el error al cuadrado.
    #   verbosity = 0    = no imprimir mensajes en pantalla.
    params <- list(
      objective        = "reg:squarederror",
      eta              = param$eta,
      max_depth        = param$max_depth,
      gamma            = param$gamma,
      colsample_bytree = param$colsample_bytree,
      min_child_weight = param$min_child_weight,
      subsample        = param$subsample,
      verbosity        = 0
    )
    
    booster <- xgboost::xgb.train(params = params, data = dtrain,
                                  nrounds = param$nrounds)
    
    # Guardamos el modelo dentro de una lista normal de R, junto con los nombres
    # de las columnas. Así evitamos modificar el objeto de xgboost (el motivo del error).
    list(booster = booster, xNames = colnames(x))
  },
  
  # CÓMO SE PREDICE: ordenamos las columnas igual que en el entrenamiento y
  # le pedimos al modelo el precio estimado de cada auto.
  predict = function(modelFit, newdata, submodels = NULL) {
    newdata <- as.matrix(newdata)[, modelFit$xNames, drop = FALSE]
    predict(modelFit$booster, newdata)
  },
  
  prob = NULL,                  # solo se usa en clasificación
  sort = function(x) x[order(x$nrounds, x$max_depth, x$eta), ]   # orden de la tabla de resultados
)

# La grilla de hiperparámetros NO cambia respecto a la versión anterior.
grid_xgb <- expand.grid(
  nrounds          = c(100, 300),  # número de árboles en la fila (cuántas rondas de corrección).
  # Pocas rondas = modelo flojo; demasiadas = memoriza los datos.
  max_depth        = c(3, 6),      # profundidad máxima de cada árbol (cuántas preguntas seguidas puede hacer).
  # 3 = árboles simples; 6 = árboles más complejos.
  eta              = c(0.05, 0.1), # "shrinkage" (tasa de aprendizaje): cuánto le creemos a cada árbol nuevo.
  # Valor chico = correcciones pequeñas y cuidadosas (más seguro pero necesita
  # más árboles); valor grande = correcciones fuertes (más rápido pero arriesgado).
  gamma            = 0,            # ganancia mínima exigida para hacer una pregunta nueva en un árbol.
  # 0 = no se exige nada extra (el árbol puede crecer libremente).
  colsample_bytree = 0.8,          # fracción de columnas que usa cada árbol: 0.8 = 80%. Es una PROPORCIÓN,
  # no una cantidad fija, así que nunca supera las columnas disponibles.
  min_child_weight = 1,            # tamaño mínimo (en "peso") que debe tener un grupo para poder dividirse.
  # 1 = permisivo; valores altos = más conservador.
  subsample        = 0.8           # fracción de filas que usa cada árbol: 0.8 = 80% de los autos, elegidos al azar.
  # Agrega variedad y reduce el riesgo de memorizar.
)

set.seed(123)
mod_xgb <- train(
  price ~ .,
  data       = train_m,
  method     = xgb_custom,  # en vez de "xgbTree": usamos la receta propia definida arriba
  metric     = "RMSE",
  trControl  = ctrl,
  tuneGrid   = grid_xgb,
  preProcess = pre
)


# ------------------------------------------------------------------------------
# 8. MODELO 4: ENSAMBLE (AVERAGING Y STACKING)
# ------------------------------------------------------------------------------

# Juntamos los 3 modelos ya entrenados en una "lista de expertos".
# (Esto es posible porque todos usaron los mismos folds y savePredictions = "final".)
lista_modelos <- as.caretList(
  list(rpart = mod_rpart, ranger = mod_ranger, xgb = mod_xgb)
)

# Hoja de instrucciones para el "meta-modelo", el que decide cómo mezclar a los expertos.
# NUEVO: también usa los mismos 5 folds (index = folds) que los modelos base.
ctrl_meta <- trainControl(
  method          = "cv",
  number          = 5,
  index           = folds,
  savePredictions = "final"
)

# --- 8a. AVERAGING PONDERADO (caretEnsemble) ----------------------------------
# Es un promedio, pero no todos los expertos pesan igual: el modelo busca los
# pesos que minimizan el error (por ejemplo 60% Random Forest, 30% XGBoost y
# 10% árbol). Se calcula usando las predicciones que cada experto hizo en sus
# "exámenes" de validación cruzada, es decir, sobre autos que NO vio al entrenar.
set.seed(123)
ens_avg <- caretEnsemble(
  lista_modelos,
  metric    = "RMSE",       # elegir los pesos que minimicen el RMSE
  trControl = ctrl_meta
)

# --- 8b. STACKING (caretStack) ------------------------------------------------
# Es un paso más sofisticado: las predicciones de los 3 expertos se usan como
# datos de entrada de un "jefe" (el meta-modelo), que aprende cuándo confiar
# más en uno u otro. Aquí el jefe es una regresión lineal ("glm"), un modelo
# simple y fácil de explicar.
set.seed(123)
ens_stack <- caretStack(
  lista_modelos,
  method    = "glm",        # tipo de meta-modelo (regresión lineal)
  metric    = "RMSE",
  trControl = ctrl_meta
)


# ------------------------------------------------------------------------------
# 9. EVALUACIÓN FINAL EN EL SET DE TEST
# ------------------------------------------------------------------------------

# Hasta ahora los modelos solo vieron datos de entrenamiento. Ahora hacemos el
# "examen final" con test, autos que nunca vieron. predict() le pide a cada
# modelo que estime el precio de esos autos; caret aplica solo el mismo
# preprocesamiento aprendido en entrenamiento.
pred_rpart  <- predict(mod_rpart,  newdata = test_m)
pred_ranger <- predict(mod_ranger, newdata = test_m)
pred_xgb    <- predict(mod_xgb,    newdata = test_m)
pred_avg    <- predict(ens_avg,    newdata = test_m)
pred_stack  <- predict(ens_stack,  newdata = test_m)

# Calculamos las métricas de cada modelo con nuestra función y las apilamos
# una debajo de otra (rbind) en una sola tabla.
resultados <- rbind(
  evaluar(pred_rpart,  test_m$price, "Árbol de regresión (rpart)"),
  evaluar(pred_ranger, test_m$price, "Random Forest (ranger)"),
  evaluar(pred_xgb,    test_m$price, "XGBoost (xgbTree)"),
  evaluar(pred_avg,    test_m$price, "Ensamble Averaging (caretEnsemble)"),
  evaluar(pred_stack,  test_m$price, "Ensamble Stacking (caretStack, glm)")
)

# Ordenamos la tabla del mejor al peor (menor RMSE arriba) y limpiamos la numeración.
resultados <- resultados[order(resultados$RMSE), ]
rownames(resultados) <- NULL

# Mostramos la tabla final. Para interpretarla: queremos RMSE y MAE lo más
# bajos posible y R² lo más cercano a 1. Para que un modelo "valga la pena",
# debe superar al mejor baseline (el promedio por marca) calculado más arriba.
print(resultados)

# Hiperparámetros ganadores de cada modelo (la configuración que obtuvo el mejor
# RMSE en la validación cruzada). Sirven para justificar las decisiones en la presentación.
mod_rpart$bestTune
mod_ranger$bestTune
mod_xgb$bestTune